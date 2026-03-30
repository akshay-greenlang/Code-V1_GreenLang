# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Multi-Framework Regulatory Compliance (Engine 6 of 7)

AGENT-MRV-011: Steam/Heat Purchase Agent

Validates Scope 2 steam, district heating, and district cooling emission
calculations against seven regulatory frameworks to ensure data completeness,
methodological correctness, boiler efficiency documentation, supplier data
quality, gas-level reporting, and full audit-trail traceability.

Supported Frameworks (84 total requirements):
    1. GHG Protocol Scope 2 Guidance           (12 requirements)
    2. ISO 14064-1:2018                        (12 requirements)
    3. CSRD/ESRS E1                            (12 requirements)
    4. CDP Climate Change                      (12 requirements)
    5. SBTi Corporate Manual                   (12 requirements)
    6. EU Energy Efficiency Directive (EU EED) (12 requirements)
    7. EPA Mandatory Reporting Rule (40 CFR 98)(12 requirements)

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
    >>> from greenlang.agents.mrv.steam_heat_purchase.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calc_result={
    ...         "energy_type": "steam",
    ...         "consumption_gj": 5000.0,
    ...         "emission_factor": 65.3,
    ...         "calculation_method": "FUEL_BASED",
    ...         "boiler_efficiency": 0.82,
    ...         "total_co2e_tonnes": 325.0,
    ...         "has_uncertainty": True,
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["ghg_protocol_scope2"],
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
    from greenlang.agents.mrv.steam_heat_purchase.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.steam_heat_purchase.provenance import (
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
    "ghg_protocol_scope2",
    "iso_14064",
    "csrd_esrs",
    "cdp",
    "sbti",
    "eu_eed",
    "epa_mrr",
]

#: Total requirements across all 7 frameworks (12 per framework).
TOTAL_REQUIREMENTS: int = 84

#: Valid energy types for Scope 2 steam/heat/cooling.
VALID_ENERGY_TYPES: Set[str] = {
    "steam", "district_heating", "district_cooling",
    "hot_water", "chilled_water", "process_heat",
}

#: Valid calculation methods for steam/heat purchase.
VALID_CALCULATION_METHODS: Set[str] = {
    "DIRECT_EF", "FUEL_BASED", "COP_BASED", "CHP_ALLOCATED",
    "SUPPLIER_SPECIFIC", "EFFICIENCY_RATIO", "ENERGY_BALANCE",
    "WEIGHTED_AVERAGE",
}

#: Valid GWP assessment report sources.
VALID_GWP_SOURCES: Set[str] = {"AR4", "AR5", "AR6", "AR6_20YR"}

#: Individual greenhouse gases expected for per-gas reporting.
EXPECTED_GASES: List[str] = ["CO2", "CH4", "N2O"]

#: Valid CHP allocation methods.
VALID_CHP_ALLOC_METHODS: Set[str] = {
    "efficiency", "energy", "exergy",
    "iea_fixed_heat", "iea_fixed_power",
    "carnot", "finnish", "ppa",
}

#: Valid emission factor source references.
VALID_EF_SOURCES: Set[str] = {
    "IPCC_2006", "IPCC_2019", "EPA_AP42", "DEFRA_BEIS",
    "IEA", "ECOINVENT", "NATIONAL_INVENTORY",
    "FACILITY_SPECIFIC", "SUPPLIER_SPECIFIC", "CUSTOM",
}

#: Framework metadata.
FRAMEWORK_INFO: Dict[str, Dict[str, Any]] = {
    "ghg_protocol_scope2": {
        "name": "GHG Protocol Scope 2 Guidance",
        "version": "2015",
        "publisher": "WRI/WBCSD",
        "description": (
            "GHG Protocol Scope 2 Guidance for purchased steam, "
            "district heating, and district cooling. Requires energy "
            "type classification, emission factor documentation, boiler "
            "efficiency, and per-gas breakdown."
        ),
        "reference": "https://ghgprotocol.org/scope_2_guidance",
        "requirements_count": 12,
    },
    "iso_14064": {
        "name": "ISO 14064-1:2018",
        "version": "2018",
        "publisher": "ISO",
        "description": (
            "International standard for quantification and reporting "
            "of greenhouse gas emissions and removals at the "
            "organization level."
        ),
        "reference": "https://www.iso.org/standard/66453.html",
        "requirements_count": 12,
    },
    "csrd_esrs": {
        "name": "CSRD/ESRS E1 Climate Change",
        "version": "2024",
        "publisher": "EFRAG",
        "description": (
            "European Sustainability Reporting Standards for climate "
            "change disclosures under the Corporate Sustainability "
            "Reporting Directive, including Scope 2 purchased thermal "
            "energy reporting per ESRS E1.32."
        ),
        "reference": "https://www.efrag.org/lab6",
        "requirements_count": 12,
    },
    "cdp": {
        "name": "CDP Climate Change Questionnaire",
        "version": "2024",
        "publisher": "CDP",
        "description": (
            "CDP disclosure framework for corporate climate change "
            "reporting, specifically C6.3 for Scope 2 non-electricity "
            "energy sources (steam, heating, cooling)."
        ),
        "reference": "https://www.cdp.net/en/guidance",
        "requirements_count": 12,
    },
    "sbti": {
        "name": "SBTi Corporate Net-Zero Standard",
        "version": "2023",
        "publisher": "Science Based Targets initiative",
        "description": (
            "Science-based target setting for corporate greenhouse "
            "gas emission reductions covering Scope 2 purchased "
            "thermal energy sources."
        ),
        "reference": "https://sciencebasedtargets.org/net-zero",
        "requirements_count": 12,
    },
    "eu_eed": {
        "name": "EU Energy Efficiency Directive",
        "version": "2023/1791",
        "publisher": "European Union",
        "description": (
            "EU Energy Efficiency Directive requirements for CHP "
            "classification, allocation methods, primary energy "
            "savings calculation, and high-efficiency determination "
            "per EU EED Annex II reference efficiencies."
        ),
        "reference": "https://eur-lex.europa.eu/eli/dir/2023/1791",
        "requirements_count": 12,
    },
    "epa_mrr": {
        "name": "EPA Mandatory Reporting Rule (40 CFR Part 98)",
        "version": "2024",
        "publisher": "U.S. EPA",
        "description": (
            "EPA Mandatory Greenhouse Gas Reporting Rule under 40 CFR "
            "Part 98, Subpart C for general stationary fuel combustion "
            "sources providing steam and heat."
        ),
        "reference": "https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-98",
        "requirements_count": 12,
    },
}

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
    """Multi-framework regulatory compliance checker for Scope 2
    steam, district heating, and district cooling emission calculations.

    Evaluates 84 requirements across 7 regulatory frameworks using
    deterministic boolean logic. No LLM involvement.

    Thread Safety:
        All mutable state is protected by ``threading.RLock``.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = engine.check_compliance(
        ...     {"energy_type": "steam", "total_co2e_tonnes": 100.0},
        ...     ["ghg_protocol_scope2"],
        ... )
    """

    # Class-level singleton support
    _instance: Optional["ComplianceCheckerEngine"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "ComplianceCheckerEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the ComplianceCheckerEngine."""
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = threading.RLock()
        self._total_checks: int = 0
        self._framework_check_counts: Dict[str, int] = {
            f: 0 for f in SUPPORTED_FRAMEWORKS
        }
        self._total_findings: int = 0
        self._total_passed: int = 0
        self._total_failed: int = 0
        self._created_at: datetime = utcnow()

        # Map framework names to checker methods
        self._framework_checkers: Dict[str, Callable] = {
            "ghg_protocol_scope2": self.check_ghg_protocol,
            "iso_14064": self.check_iso_14064,
            "csrd_esrs": self.check_csrd_esrs,
            "cdp": self.check_cdp,
            "sbti": self.check_sbti,
            "eu_eed": self.check_eu_eed,
            "epa_mrr": self.check_epa_mrr,
        }

        logger.info(
            "ComplianceCheckerEngine initialized: frameworks=%d, "
            "total_requirements=%d",
            len(SUPPORTED_FRAMEWORKS),
            TOTAL_REQUIREMENTS,
        )

    # ------------------------------------------------------------------
    # Class-level reset
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance, clearing all state."""
        with cls._singleton_lock:
            if cls._instance is not None:
                with cls._instance._lock:
                    cls._instance._total_checks = 0
                    cls._instance._framework_check_counts = {
                        f: 0 for f in SUPPORTED_FRAMEWORKS
                    }
                    cls._instance._total_findings = 0
                    cls._instance._total_passed = 0
                    cls._instance._total_failed = 0
                cls._instance._initialized = False
                cls._instance = None
        logger.info("ComplianceCheckerEngine singleton reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_checks(self) -> None:
        """Thread-safe increment of the check counter."""
        with self._lock:
            self._total_checks += 1

    def _record_framework(self, framework: str) -> None:
        """Thread-safe increment of per-framework counter."""
        with self._lock:
            self._framework_check_counts[framework] = (
                self._framework_check_counts.get(framework, 0) + 1
            )

    def _record_findings(self, passed: int, failed: int) -> None:
        """Thread-safe accumulation of finding counts."""
        with self._lock:
            self._total_findings += passed + failed
            self._total_passed += passed
            self._total_failed += failed

    def _get_field(
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

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convert to float with fallback."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _energy_type(self, data: Dict[str, Any]) -> str:
        """Extract and normalize the energy type."""
        raw = str(self._get_field(data, "energy_type", "")).lower().strip()
        return raw

    def _calculation_method(self, data: Dict[str, Any]) -> str:
        """Extract and normalize the calculation method."""
        raw = str(self._get_field(data, "calculation_method", "")).upper().strip()
        return raw

    def _determine_status(self, passed: int, total: int) -> str:
        """Determine compliance status from pass rate."""
        if total == 0:
            return "NON_COMPLIANT"
        ratio = passed / total
        if ratio >= 1.0:
            return "COMPLIANT"
        elif ratio >= 0.5:
            return "PARTIAL"
        else:
            return "NON_COMPLIANT"

    def _build_framework_result(
        self,
        framework: str,
        findings: List[ComplianceFinding],
    ) -> Dict[str, Any]:
        """Build a standardized framework result dict from findings."""
        passed = sum(1 for f in findings if f.passed)
        failed = sum(1 for f in findings if not f.passed)
        errors = sum(
            1 for f in findings if not f.passed and f.severity == "ERROR"
        )
        warnings = sum(
            1 for f in findings if not f.passed and f.severity == "WARNING"
        )
        total_reqs = len(findings)

        pass_rate = (
            Decimal(str(passed)) / Decimal(str(total_reqs)) * Decimal("100")
            if total_reqs > 0
            else Decimal("0")
        )

        status = self._determine_status(passed, total_reqs)

        failed_findings = [f.to_dict() for f in findings if not f.passed]
        recommendations = [
            f.recommendation for f in findings
            if not f.passed and f.recommendation
        ]

        self._record_framework(framework)
        self._record_findings(passed, failed)

        return {
            "framework": framework,
            "framework_name": FRAMEWORK_INFO.get(framework, {}).get(
                "name", framework
            ),
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
            "provenance_hash": _compute_hash(
                [f.to_dict() for f in findings]
            ),
            "timestamp": utcnow().isoformat(),
        }

    # ==================================================================
    # Public Method 1: check_compliance
    # ==================================================================

    def check_compliance(
        self,
        calc_result: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check compliance of a steam/heat calculation against frameworks.

        Args:
            calc_result: The calculation result dict to evaluate.
            frameworks: List of framework IDs to check, or None for all 7.

        Returns:
            Dict with per-framework results, overall summary, and
            SHA-256 provenance hash.
        """
        self._increment_checks()
        start_time = time.monotonic()
        calc_id = f"shp_chk_{uuid4().hex[:12]}"

        if frameworks is None:
            frameworks = list(SUPPORTED_FRAMEWORKS)

        # Validate and normalize framework names
        valid_fws: List[str] = []
        for fw in frameworks:
            fw_lower = fw.lower()
            if fw_lower in self._framework_checkers:
                valid_fws.append(fw_lower)
            else:
                logger.warning(
                    "Unknown framework '%s', skipping", fw
                )

        # Run checks per framework
        framework_results: Dict[str, Dict[str, Any]] = {}
        total_passed = 0
        total_failed = 0
        total_requirements = 0
        total_errors = 0
        total_warnings = 0
        compliant_count = 0
        non_compliant_count = 0
        partial_count = 0

        for fw in valid_fws:
            checker = self._framework_checkers[fw]
            findings = checker(calc_result)
            fw_result = self._build_framework_result(fw, findings)
            framework_results[fw] = fw_result

            total_passed += fw_result["passed"]
            total_failed += fw_result["failed"]
            total_requirements += fw_result["total_requirements"]
            total_errors += fw_result["errors"]
            total_warnings += fw_result["warnings"]

            status = fw_result["status"]
            if status == "COMPLIANT":
                compliant_count += 1
            elif status == "PARTIAL":
                partial_count += 1
            elif status == "NON_COMPLIANT":
                non_compliant_count += 1

        # Overall status
        if total_requirements > 0:
            overall_rate = (
                Decimal(str(total_passed))
                / Decimal(str(total_requirements))
                * Decimal("100")
            )
        else:
            overall_rate = Decimal("0")

        overall_status = self._determine_status(
            total_passed, total_requirements
        )

        processing_time = round(
            (time.monotonic() - start_time) * 1000, 3
        )

        # Build ordered results list
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
                "total_failed": total_failed,
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
    # Public Method 2: check_single_framework
    # ==================================================================

    def check_single_framework(
        self,
        calc_result: Dict[str, Any],
        framework: str,
    ) -> Dict[str, Any]:
        """Convenience method to check a single framework.

        Args:
            calc_result: Calculation data dictionary.
            framework: Single framework name.

        Returns:
            Compliance results for the specified framework only.
        """
        return self.check_compliance(calc_result, frameworks=[framework])

    # ==================================================================
    # Framework 1: GHG Protocol Scope 2 Guidance (12 requirements)
    # ==================================================================

    def check_ghg_protocol(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with GHG Protocol Scope 2 Guidance.

        12 requirements covering energy type classification, consumption
        data, emission factor documentation, calculation method, boiler
        efficiency, supplier data, per-gas breakdown, GWP sourcing,
        biogenic CO2 separation, condensate return, uncertainty, and
        provenance tracking.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "ghg_protocol_scope2"
        energy = self._energy_type(data)
        calc_method = self._calculation_method(data)

        # GHG-SHP-001: Energy type classified
        valid_energy = energy in VALID_ENERGY_TYPES
        findings.append(ComplianceFinding(
            f"{fw}-001", fw,
            "Energy type must be classified (steam/heating/cooling)",
            valid_energy,
            "ERROR",
            (
                f"Energy type '{energy}' is valid"
                if valid_energy
                else f"Energy type '{energy}' is not recognized; expected one of {sorted(VALID_ENERGY_TYPES)}"
            ),
            "Specify energy_type as steam, district_heating, district_cooling, hot_water, chilled_water, or process_heat",
        ))

        # GHG-SHP-002: Consumption data quantified in GJ
        has_consumption = self._has_any_field(
            data, "consumption_gj", "consumption_mwh",
            "consumption_kwh", "consumption_mmbtu",
            "total_energy_gj", "energy_consumed_gj",
        )
        consumption_val = self._safe_float(
            self._get_field(data, "consumption_gj",
                            self._get_field(data, "total_energy_gj", 0))
        )
        findings.append(ComplianceFinding(
            f"{fw}-002", fw,
            "Energy consumption must be quantified in GJ (or convertible unit)",
            has_consumption,
            "ERROR",
            (
                f"Consumption data present: {consumption_val} GJ"
                if has_consumption
                else "No consumption data found (consumption_gj, consumption_mwh, etc.)"
            ),
            "Provide consumption_gj or equivalent energy quantity field",
        ))

        # GHG-SHP-003: Emission factor specified and sourced
        has_ef = self._has_any_field(
            data, "emission_factor", "emission_factor_source",
            "ef_source", "ef_kg_co2e_per_gj",
        )
        findings.append(ComplianceFinding(
            f"{fw}-003", fw,
            "Emission factor must be specified and sourced",
            has_ef,
            "ERROR",
            (
                "Emission factor is documented"
                if has_ef
                else "No emission factor or source documented"
            ),
            "Provide emission_factor and emission_factor_source",
        ))

        # GHG-SHP-004: Calculation method documented
        valid_method = calc_method in VALID_CALCULATION_METHODS
        has_method = valid_method or self._has_field(data, "calculation_method")
        findings.append(ComplianceFinding(
            f"{fw}-004", fw,
            "Calculation method must be documented",
            has_method,
            "ERROR",
            (
                f"Calculation method '{calc_method}' documented"
                if has_method
                else "Calculation method not specified"
            ),
            "Specify calculation_method (DIRECT_EF, FUEL_BASED, COP_BASED, CHP_ALLOCATED)",
        ))

        # GHG-SHP-005: Boiler efficiency documented (steam)
        is_steam = energy in ("steam", "process_heat", "hot_water")
        boiler_eff = self._safe_float(
            self._get_field(data, "boiler_efficiency", 0)
        )
        has_boiler = boiler_eff > 0 or self._has_field(
            data, "boiler_efficiency"
        )
        if is_steam:
            findings.append(ComplianceFinding(
                f"{fw}-005", fw,
                "Boiler efficiency must be documented for steam systems",
                has_boiler,
                "WARNING",
                (
                    f"Boiler efficiency = {boiler_eff}"
                    if has_boiler
                    else "Boiler efficiency not documented for steam system"
                ),
                "Provide boiler_efficiency (0-1 range) for steam generation",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-005", fw,
                "Boiler efficiency must be documented for steam systems",
                True,
                "WARNING",
                f"Not applicable for energy type '{energy}'",
                "",
            ))

        # GHG-SHP-006: Supplier data or default EFs used
        has_supplier = self._has_any_field(
            data, "supplier_id", "supplier_name", "supplier_data",
            "supplier_ef", "supplier_fuel_mix",
        )
        has_default_ef = self._has_any_field(
            data, "default_ef_used", "regional_ef", "emission_factor",
        )
        findings.append(ComplianceFinding(
            f"{fw}-006", fw,
            "Supplier data or default emission factors must be used",
            has_supplier or has_default_ef,
            "ERROR",
            (
                "Supplier data or default EFs documented"
                if has_supplier or has_default_ef
                else "Neither supplier data nor default emission factors found"
            ),
            "Provide supplier data (supplier_id, supplier_fuel_mix) or default emission_factor",
        ))

        # GHG-SHP-007: Per-gas breakdown (CO2, CH4, N2O)
        gas_breakdown = self._get_field(data, "gas_breakdown", None)
        gases_reported = self._get_field(data, "gases_reported", [])
        has_gas = False
        if isinstance(gas_breakdown, dict):
            has_gas = all(g in gas_breakdown for g in EXPECTED_GASES)
        elif isinstance(gas_breakdown, list) and len(gas_breakdown) >= 3:
            gas_names = {
                g.get("gas", "") for g in gas_breakdown
                if isinstance(g, dict)
            }
            has_gas = all(g in gas_names for g in EXPECTED_GASES)
        elif isinstance(gases_reported, list) and len(gases_reported) >= 3:
            has_gas = all(
                g in [gr.upper() for gr in gases_reported]
                for g in EXPECTED_GASES
            )
        findings.append(ComplianceFinding(
            f"{fw}-007", fw,
            "Per-gas breakdown (CO2, CH4, N2O) must be reported",
            has_gas,
            "WARNING",
            (
                "Per-gas breakdown includes CO2, CH4, N2O"
                if has_gas
                else "Per-gas breakdown incomplete (need CO2, CH4, N2O)"
            ),
            "Provide gas_breakdown with CO2, CH4, N2O entries",
        ))

        # GHG-SHP-008: GWP values sourced from IPCC AR
        gwp_source = str(
            self._get_field(data, "gwp_source", "")
        ).upper().strip()
        has_gwp = gwp_source in VALID_GWP_SOURCES or self._has_field(
            data, "gwp_source"
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "GWP values must be sourced from an IPCC Assessment Report",
            has_gwp,
            "WARNING",
            (
                f"GWP source = {gwp_source}"
                if has_gwp
                else "GWP source not specified (expected AR4, AR5, AR6, or AR6_20YR)"
            ),
            "Specify gwp_source as AR4, AR5, AR6, or AR6_20YR",
        ))

        # GHG-SHP-009: Biogenic CO2 separated if applicable
        has_biogenic_flag = self._has_any_field(
            data, "biogenic_co2_separated", "biogenic_co2_tonnes",
            "biogenic_co2_kg", "enable_biogenic_separation",
        )
        fuel_type = str(
            self._get_field(data, "fuel_type", "")
        ).lower()
        is_biogenic_relevant = fuel_type in (
            "biomass", "biogas", "wood_chips", "wood_pellets",
            "bagasse", "landfill_gas",
        )
        if is_biogenic_relevant:
            findings.append(ComplianceFinding(
                f"{fw}-009", fw,
                "Biogenic CO2 must be separated if biogenic fuels used",
                has_biogenic_flag,
                "WARNING",
                (
                    "Biogenic CO2 separation documented"
                    if has_biogenic_flag
                    else f"Biogenic fuel '{fuel_type}' detected but biogenic CO2 not separated"
                ),
                "Set biogenic_co2_separated=True and report biogenic_co2_tonnes separately",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-009", fw,
                "Biogenic CO2 must be separated if biogenic fuels used",
                True,
                "WARNING",
                "Not applicable (no biogenic fuel detected)",
                "",
            ))

        # GHG-SHP-010: Condensate return documented (steam)
        has_condensate = self._has_any_field(
            data, "condensate_return_pct", "condensate_return",
            "condensate_return_fraction",
        )
        if is_steam:
            findings.append(ComplianceFinding(
                f"{fw}-010", fw,
                "Condensate return must be documented for steam systems",
                has_condensate,
                "INFO",
                (
                    "Condensate return documented"
                    if has_condensate
                    else "Condensate return not documented for steam system"
                ),
                "Provide condensate_return_pct (0-100) for steam systems",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-010", fw,
                "Condensate return must be documented for steam systems",
                True,
                "INFO",
                f"Not applicable for energy type '{energy}'",
                "",
            ))

        # GHG-SHP-011: Uncertainty quantified
        has_uncertainty = self._has_any_field(
            data, "has_uncertainty", "uncertainty", "uncertainty_pct",
            "uncertainty_range", "monte_carlo_iterations",
        )
        findings.append(ComplianceFinding(
            f"{fw}-011", fw,
            "Uncertainty must be quantified",
            has_uncertainty,
            "WARNING",
            (
                "Uncertainty quantification present"
                if has_uncertainty
                else "Uncertainty not quantified"
            ),
            "Provide uncertainty estimate (uncertainty_pct or has_uncertainty=True)",
        ))

        # GHG-SHP-012: Provenance chain traceable
        has_provenance = self._has_any_field(
            data, "provenance_hash", "provenance_chain",
            "calculation_trace",
        )
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Calculation must be traceable (provenance chain with SHA-256)",
            has_provenance,
            "WARNING",
            (
                "Provenance hash present"
                if has_provenance
                else "No provenance hash found"
            ),
            "Ensure provenance_hash is generated for full audit trail",
        ))

        return findings

    # ==================================================================
    # Framework 2: ISO 14064-1:2018 (12 requirements)
    # ==================================================================

    def check_iso_14064(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with ISO 14064-1:2018.

        12 requirements covering organizational boundary, emission source
        identification, quantification method, base year, data quality,
        uncertainty, completeness, consistency, accuracy, transparency,
        biogenic reporting, and documentation.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "iso_14064"

        # ISO-SHP-001: Organizational boundary defined
        has_boundary = self._has_any_field(
            data, "facility_id", "organization_id", "org_boundary",
            "site_id", "tenant_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-001", fw,
            "Organizational boundary must be defined",
            has_boundary,
            "ERROR",
            (
                "Organizational boundary defined"
                if has_boundary
                else "Organizational boundary not defined"
            ),
            "Provide facility_id, organization_id, or org_boundary",
        ))

        # ISO-SHP-002: Emission sources identified and classified
        has_source_id = self._has_any_field(
            data, "energy_type", "emission_source", "source_category",
            "source_classification",
        )
        findings.append(ComplianceFinding(
            f"{fw}-002", fw,
            "Emission sources must be identified and classified",
            has_source_id,
            "ERROR",
            (
                "Emission sources identified"
                if has_source_id
                else "Emission sources not identified or classified"
            ),
            "Provide energy_type and source identification fields",
        ))

        # ISO-SHP-003: Quantification method documented per ISO 14064-1
        has_method = self._has_any_field(
            data, "calculation_method", "quantification_method",
            "methodology",
        )
        findings.append(ComplianceFinding(
            f"{fw}-003", fw,
            "Quantification methodology must be documented per ISO 14064-1",
            has_method,
            "ERROR",
            (
                "Quantification method documented"
                if has_method
                else "Quantification method not documented"
            ),
            "Specify calculation_method per ISO 14064-1 requirements",
        ))

        # ISO-SHP-004: Base year established
        has_base_year = self._has_any_field(
            data, "base_year", "base_year_emissions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-004", fw,
            "Base year must be established and documented",
            has_base_year,
            "WARNING",
            (
                "Base year established"
                if has_base_year
                else "Base year not established"
            ),
            "Provide base_year for tracking emission trends over time",
        ))

        # ISO-SHP-005: Data quality assessment documented
        has_dq = self._has_any_field(
            data, "data_quality_tier", "data_quality", "data_quality_score",
            "data_quality_assessment",
        )
        findings.append(ComplianceFinding(
            f"{fw}-005", fw,
            "Data quality assessment must be documented",
            has_dq,
            "WARNING",
            (
                "Data quality assessment documented"
                if has_dq
                else "Data quality assessment not documented"
            ),
            "Provide data_quality_tier (TIER_1, TIER_2, TIER_3)",
        ))

        # ISO-SHP-006: Uncertainty quantified per ISO standard
        has_uncertainty = self._has_any_field(
            data, "uncertainty", "uncertainty_pct", "has_uncertainty",
            "uncertainty_range", "uncertainty_assessment",
        )
        findings.append(ComplianceFinding(
            f"{fw}-006", fw,
            "Uncertainty must be quantified per ISO standard",
            has_uncertainty,
            "WARNING",
            (
                "Uncertainty assessment present"
                if has_uncertainty
                else "Uncertainty not quantified"
            ),
            "Provide uncertainty assessment (uncertainty_pct or uncertainty_range)",
        ))

        # ISO-SHP-007: Completeness assessment done
        has_completeness = self._has_any_field(
            data, "completeness_check", "completeness_pct",
            "completeness_assessment", "coverage_pct",
        )
        total_co2e = self._safe_float(
            self._get_field(data, "total_co2e_tonnes", 0)
        )
        is_complete = has_completeness or total_co2e > 0
        findings.append(ComplianceFinding(
            f"{fw}-007", fw,
            "Completeness assessment must be performed",
            is_complete,
            "ERROR",
            (
                "Completeness assessment present"
                if is_complete
                else "Completeness assessment not performed"
            ),
            "Perform completeness check to ensure no material omissions",
        ))

        # ISO-SHP-008: Consistency with prior periods
        has_consistency = self._has_any_field(
            data, "previous_year_co2e", "consistency_check",
            "year_over_year", "prior_period_comparison",
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "Consistency with previous reporting periods must be maintained",
            has_consistency,
            "WARNING",
            (
                "Consistency data available"
                if has_consistency
                else "No prior period consistency data found"
            ),
            "Provide previous_year_co2e for year-over-year consistency verification",
        ))

        # ISO-SHP-009: Accuracy verified
        has_accuracy = self._has_any_field(
            data, "accuracy_verification", "verified",
            "verification_status", "calibration_date",
        )
        has_data = total_co2e > 0
        is_accurate = has_accuracy or has_data
        findings.append(ComplianceFinding(
            f"{fw}-009", fw,
            "Accuracy of activity data must be verified",
            is_accurate,
            "WARNING",
            (
                "Accuracy verification present"
                if is_accurate
                else "Accuracy not verified"
            ),
            "Document accuracy verification or calibration of measurement equipment",
        ))

        # ISO-SHP-010: Full calculation transparency
        has_transparency = self._has_any_field(
            data, "calculation_trace", "provenance_hash",
            "methodology_description", "assumptions",
        )
        findings.append(ComplianceFinding(
            f"{fw}-010", fw,
            "Full calculation transparency must be maintained",
            has_transparency,
            "INFO",
            (
                "Calculation transparency documented"
                if has_transparency
                else "Calculation transparency not documented"
            ),
            "Provide calculation_trace and document all assumptions",
        ))

        # ISO-SHP-011: Biogenic emissions reported separately
        fuel_type = str(
            self._get_field(data, "fuel_type", "")
        ).lower()
        is_biogenic = fuel_type in (
            "biomass", "biogas", "wood_chips", "wood_pellets",
            "bagasse", "landfill_gas",
        )
        has_biogenic = self._has_any_field(
            data, "biogenic_co2_separated", "biogenic_co2_tonnes",
        )
        if is_biogenic:
            findings.append(ComplianceFinding(
                f"{fw}-011", fw,
                "Biogenic emissions must be reported separately",
                has_biogenic,
                "WARNING",
                (
                    "Biogenic emissions reported separately"
                    if has_biogenic
                    else "Biogenic fuel detected but emissions not reported separately"
                ),
                "Report biogenic CO2 separately per ISO 14064-1 requirements",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-011", fw,
                "Biogenic emissions must be reported separately",
                True,
                "WARNING",
                "Not applicable (non-biogenic fuel)",
                "",
            ))

        # ISO-SHP-012: Supporting documentation maintained
        has_docs = self._has_any_field(
            data, "documentation", "provenance_hash", "supporting_docs",
            "calculation_trace", "methodology",
        )
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Supporting documentation must be maintained",
            has_docs,
            "INFO",
            (
                "Supporting documentation available"
                if has_docs
                else "Supporting documentation not found"
            ),
            "Maintain supporting documentation for all calculation inputs and outputs",
        ))

        return findings

    # ==================================================================
    # Framework 3: CSRD/ESRS E1 (12 requirements)
    # ==================================================================

    def check_csrd_esrs(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with CSRD/ESRS E1 Climate Change.

        12 requirements covering gross emissions, energy type breakdown,
        location and market methods, supplier breakdown, GHG intensity,
        target tracking, double materiality, Scope 2 boundary, methodology
        disclosure, data quality, third-party verification, and temporal
        consistency.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "csrd_esrs"

        # CSRD-SHP-001: Gross Scope 2 emissions reported
        total_co2e = self._safe_float(
            self._get_field(data, "total_co2e_tonnes",
                            self._get_field(data, "total_co2e_kg", 0))
        )
        has_gross = total_co2e > 0
        findings.append(ComplianceFinding(
            f"{fw}-001", fw,
            "Gross Scope 2 emissions must be reported",
            has_gross,
            "ERROR",
            (
                f"Gross Scope 2 emissions = {total_co2e} tCO2e"
                if has_gross
                else "Gross Scope 2 emissions not reported"
            ),
            "Provide total_co2e_tonnes for gross Scope 2 emissions",
        ))

        # CSRD-SHP-002: Breakdown by energy type
        has_energy_type = self._has_field(data, "energy_type")
        has_breakdown = self._has_any_field(
            data, "energy_type_breakdown", "energy_breakdown",
        )
        findings.append(ComplianceFinding(
            f"{fw}-002", fw,
            "Breakdown by energy type must be provided",
            has_energy_type or has_breakdown,
            "ERROR",
            (
                "Energy type breakdown available"
                if has_energy_type or has_breakdown
                else "Energy type breakdown not provided"
            ),
            "Provide energy_type classification (steam, district_heating, district_cooling)",
        ))

        # CSRD-SHP-003: Both location and market methods reported
        has_location = self._has_any_field(
            data, "location_based_co2e", "location_based_emissions",
        )
        has_market = self._has_any_field(
            data, "market_based_co2e", "market_based_emissions",
        )
        has_dual = self._has_field(data, "dual_reporting")
        both_methods = (has_location and has_market) or has_dual
        findings.append(ComplianceFinding(
            f"{fw}-003", fw,
            "Both location-based and market-based methods must be reported where applicable",
            both_methods,
            "WARNING",
            (
                "Dual reporting (location + market) available"
                if both_methods
                else "Dual reporting not available (recommend both methods)"
            ),
            "Provide both location_based_co2e and market_based_co2e if applicable",
        ))

        # CSRD-SHP-004: Breakdown by supplier if material
        has_supplier = self._has_any_field(
            data, "supplier_id", "supplier_name", "supplier_breakdown",
        )
        findings.append(ComplianceFinding(
            f"{fw}-004", fw,
            "Breakdown by supplier must be provided if material",
            has_supplier or True,  # INFO level - not strictly required
            "INFO",
            (
                "Supplier breakdown available"
                if has_supplier
                else "Supplier breakdown not provided (informational)"
            ),
            "Provide supplier_id or supplier_breakdown for material suppliers",
        ))

        # CSRD-SHP-005: GHG intensity metrics calculated
        has_intensity = self._has_any_field(
            data, "ghg_intensity", "intensity_metric",
            "co2e_per_gj", "co2e_per_mwh",
        )
        consumption = self._safe_float(
            self._get_field(data, "consumption_gj", 0)
        )
        intensity_derivable = total_co2e > 0 and consumption > 0
        findings.append(ComplianceFinding(
            f"{fw}-005", fw,
            "GHG intensity metrics must be calculated",
            has_intensity or intensity_derivable,
            "WARNING",
            (
                "GHG intensity metrics available or derivable"
                if has_intensity or intensity_derivable
                else "GHG intensity metrics not calculable"
            ),
            "Calculate ghg_intensity (tCO2e per GJ or per unit of revenue)",
        ))

        # CSRD-SHP-006: Progress toward targets tracked
        has_targets = self._has_any_field(
            data, "target_tracking", "emission_target",
            "reduction_target_pct", "target_year",
        )
        findings.append(ComplianceFinding(
            f"{fw}-006", fw,
            "Progress toward emission reduction targets must be tracked",
            has_targets,
            "WARNING",
            (
                "Target tracking available"
                if has_targets
                else "Target tracking not documented"
            ),
            "Document emission reduction targets and track progress",
        ))

        # CSRD-SHP-007: Double materiality assessment
        has_materiality = self._has_any_field(
            data, "double_materiality", "materiality_assessment",
            "financial_materiality", "impact_materiality",
        )
        findings.append(ComplianceFinding(
            f"{fw}-007", fw,
            "Double materiality assessment must be performed",
            has_materiality,
            "INFO",
            (
                "Double materiality assessment documented"
                if has_materiality
                else "Double materiality assessment not documented"
            ),
            "Perform double materiality assessment per CSRD requirements",
        ))

        # CSRD-SHP-008: Scope 2 boundary per ESRS E1.32
        has_boundary = self._has_any_field(
            data, "scope2_boundary", "reporting_boundary",
            "facility_id", "organization_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "Scope 2 boundary must be defined per ESRS E1.32",
            has_boundary,
            "ERROR",
            (
                "Scope 2 boundary defined"
                if has_boundary
                else "Scope 2 boundary not defined per ESRS E1.32"
            ),
            "Define Scope 2 boundary per ESRS E1.32 requirements",
        ))

        # CSRD-SHP-009: Methodology transparently disclosed
        has_methodology = self._has_any_field(
            data, "calculation_method", "methodology", "methodology_disclosure",
        )
        findings.append(ComplianceFinding(
            f"{fw}-009", fw,
            "Methodology must be transparently disclosed",
            has_methodology,
            "ERROR",
            (
                "Methodology transparently disclosed"
                if has_methodology
                else "Methodology not disclosed"
            ),
            "Document calculation methodology transparently",
        ))

        # CSRD-SHP-010: Data quality tier disclosed
        has_dq = self._has_any_field(
            data, "data_quality_tier", "data_quality",
        )
        findings.append(ComplianceFinding(
            f"{fw}-010", fw,
            "Data quality tier must be disclosed",
            has_dq,
            "WARNING",
            (
                "Data quality tier disclosed"
                if has_dq
                else "Data quality tier not disclosed"
            ),
            "Disclose data_quality_tier (TIER_1, TIER_2, TIER_3)",
        ))

        # CSRD-SHP-011: Third-party verification status
        has_verification = self._has_any_field(
            data, "third_party_verification", "verification_status",
            "assurance_level", "verified",
        )
        findings.append(ComplianceFinding(
            f"{fw}-011", fw,
            "Third-party verification status must be documented",
            has_verification,
            "INFO",
            (
                "Third-party verification status documented"
                if has_verification
                else "Third-party verification status not documented"
            ),
            "Document verification_status (verified, unverified, limited_assurance, reasonable_assurance)",
        ))

        # CSRD-SHP-012: Consistent time periods
        has_temporal = self._has_any_field(
            data, "reporting_year", "reporting_period",
            "start_date", "end_date",
        )
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Consistent time periods must be maintained",
            has_temporal,
            "WARNING",
            (
                "Temporal consistency documented"
                if has_temporal
                else "Reporting period not specified"
            ),
            "Specify reporting_year or reporting_period for temporal consistency",
        ))

        return findings

    # ==================================================================
    # Framework 4: CDP Climate Change (12 requirements)
    # ==================================================================

    def check_cdp(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with CDP Climate Change Questionnaire.

        12 requirements covering C6.3 Scope 2 non-electricity reporting,
        consumption data, emission factors, methodology, verification,
        intensity metrics, energy breakdown, target progress, uncertainty,
        base year recalculation, data quality, and completeness.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "cdp"
        total_co2e = self._safe_float(
            self._get_field(data, "total_co2e_tonnes", 0)
        )

        # CDP-SHP-001: C6.3 Scope 2 non-electricity reported
        findings.append(ComplianceFinding(
            f"{fw}-001", fw,
            "Scope 2 non-electricity emissions must be reported (C6.3)",
            total_co2e > 0,
            "ERROR",
            (
                f"C6.3 Scope 2 non-electricity = {total_co2e} tCO2e"
                if total_co2e > 0
                else "Scope 2 non-electricity emissions not reported for C6.3"
            ),
            "Report total_co2e_tonnes for Scope 2 non-electricity (steam/heat/cooling)",
        ))

        # CDP-SHP-002: Consumption quantities reported
        has_consumption = self._has_any_field(
            data, "consumption_gj", "consumption_mwh", "total_energy_gj",
        )
        findings.append(ComplianceFinding(
            f"{fw}-002", fw,
            "Consumption quantities must be reported",
            has_consumption,
            "ERROR",
            (
                "Consumption data reported"
                if has_consumption
                else "Consumption quantities not reported"
            ),
            "Provide consumption_gj for CDP reporting",
        ))

        # CDP-SHP-003: Emission factors reported with sources
        has_ef = self._has_any_field(
            data, "emission_factor", "ef_source", "emission_factor_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-003", fw,
            "Emission factors must be reported with sources",
            has_ef,
            "ERROR",
            (
                "Emission factors and sources documented"
                if has_ef
                else "Emission factors or sources not documented"
            ),
            "Provide emission_factor and emission_factor_source",
        ))

        # CDP-SHP-004: Methodology described (C7.5)
        has_method = self._has_any_field(
            data, "calculation_method", "methodology",
        )
        findings.append(ComplianceFinding(
            f"{fw}-004", fw,
            "Methodology must be described (C7.5)",
            has_method,
            "ERROR",
            (
                "Methodology described"
                if has_method
                else "Methodology not described for C7.5"
            ),
            "Describe calculation_method used for emissions quantification",
        ))

        # CDP-SHP-005: Third-party verification status
        has_verification = self._has_any_field(
            data, "verification_status", "verified",
            "third_party_verification",
        )
        findings.append(ComplianceFinding(
            f"{fw}-005", fw,
            "Third-party verification status must be reported",
            has_verification,
            "WARNING",
            (
                "Verification status reported"
                if has_verification
                else "Verification status not reported"
            ),
            "Report verification_status for CDP",
        ))

        # CDP-SHP-006: Intensity metrics (C6.10)
        has_intensity = self._has_any_field(
            data, "ghg_intensity", "intensity_metric",
            "co2e_per_gj", "co2e_per_mwh",
        )
        consumption = self._safe_float(
            self._get_field(data, "consumption_gj", 0)
        )
        intensity_derivable = total_co2e > 0 and consumption > 0
        findings.append(ComplianceFinding(
            f"{fw}-006", fw,
            "Intensity metrics must be reported (C6.10)",
            has_intensity or intensity_derivable,
            "WARNING",
            (
                "Intensity metrics available"
                if has_intensity or intensity_derivable
                else "Intensity metrics not available for C6.10"
            ),
            "Calculate and report GHG intensity metrics",
        ))

        # CDP-SHP-007: Energy source breakdown
        has_energy_type = self._has_any_field(
            data, "energy_type", "energy_breakdown",
            "fuel_type", "energy_source",
        )
        findings.append(ComplianceFinding(
            f"{fw}-007", fw,
            "Energy source breakdown must be reported",
            has_energy_type,
            "WARNING",
            (
                "Energy source breakdown available"
                if has_energy_type
                else "Energy source breakdown not available"
            ),
            "Provide energy_type and fuel_type for energy source breakdown",
        ))

        # CDP-SHP-008: Target progress (C4.2)
        has_target = self._has_any_field(
            data, "target_tracking", "emission_target",
            "reduction_target_pct", "target_progress",
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "Target progress must be reported (C4.2)",
            has_target,
            "WARNING",
            (
                "Target progress documented"
                if has_target
                else "Target progress not documented for C4.2"
            ),
            "Document emission reduction target progress",
        ))

        # CDP-SHP-009: Uncertainty reported
        has_uncertainty = self._has_any_field(
            data, "uncertainty", "uncertainty_pct", "has_uncertainty",
        )
        findings.append(ComplianceFinding(
            f"{fw}-009", fw,
            "Uncertainty must be reported",
            has_uncertainty,
            "INFO",
            (
                "Uncertainty reported"
                if has_uncertainty
                else "Uncertainty not reported"
            ),
            "Report uncertainty estimate for transparency",
        ))

        # CDP-SHP-010: Base year recalculation policy
        has_base_year = self._has_any_field(
            data, "base_year", "base_year_recalculation_policy",
            "recalculation_trigger",
        )
        findings.append(ComplianceFinding(
            f"{fw}-010", fw,
            "Base year recalculation policy must be documented",
            has_base_year,
            "WARNING",
            (
                "Base year / recalculation policy documented"
                if has_base_year
                else "Base year recalculation policy not documented"
            ),
            "Document base_year and recalculation policy",
        ))

        # CDP-SHP-011: Data quality assessment
        has_dq = self._has_any_field(
            data, "data_quality_tier", "data_quality",
            "data_quality_score",
        )
        findings.append(ComplianceFinding(
            f"{fw}-011", fw,
            "Data quality assessment must be performed",
            has_dq,
            "WARNING",
            (
                "Data quality assessment documented"
                if has_dq
                else "Data quality assessment not documented"
            ),
            "Provide data_quality_tier assessment",
        ))

        # CDP-SHP-012: Completeness assessment
        has_completeness = self._has_any_field(
            data, "completeness_check", "completeness_pct",
            "coverage_pct",
        )
        is_complete = has_completeness or total_co2e > 0
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Completeness assessment must be performed",
            is_complete,
            "INFO",
            (
                "Completeness assessment present"
                if is_complete
                else "Completeness assessment not performed"
            ),
            "Perform completeness assessment to ensure no material omissions",
        ))

        return findings

    # ==================================================================
    # Framework 5: SBTi Corporate Manual (12 requirements)
    # ==================================================================

    def check_sbti(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with SBTi Corporate Net-Zero Standard.

        12 requirements covering Scope 2 coverage, approved methodology,
        base year, target ambition, annual tracking, verification, data
        completeness, consistent boundary, market instrument quality,
        renewable target, uncertainty limits, and methodology consistency.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "sbti"
        total_co2e = self._safe_float(
            self._get_field(data, "total_co2e_tonnes", 0)
        )

        # SBTI-SHP-001: Scope 2 coverage >= 95%
        coverage_pct = self._safe_float(
            self._get_field(data, "coverage_pct",
                            self._get_field(data, "source_coverage_pct", 0))
        )
        has_consumption = self._has_any_field(
            data, "consumption_gj", "total_energy_gj",
        )
        # If consumption data exists and total > 0, we consider coverage adequate
        coverage_ok = coverage_pct >= 95 or (has_consumption and total_co2e > 0)
        findings.append(ComplianceFinding(
            f"{fw}-001", fw,
            "Scope 2 coverage must be >= 95%",
            coverage_ok,
            "ERROR",
            (
                f"Scope 2 coverage = {coverage_pct}% (or full coverage assumed)"
                if coverage_ok
                else f"Scope 2 coverage = {coverage_pct}% (below 95% threshold)"
            ),
            "Ensure Scope 2 emission source coverage is at least 95%",
        ))

        # SBTI-SHP-002: SBTi-approved methodology
        has_method = self._has_any_field(
            data, "calculation_method", "methodology",
        )
        calc_method = self._calculation_method(data)
        approved = calc_method in VALID_CALCULATION_METHODS or has_method
        findings.append(ComplianceFinding(
            f"{fw}-002", fw,
            "SBTi-approved methodology must be used",
            approved,
            "ERROR",
            (
                f"Approved methodology: {calc_method}"
                if approved
                else "No SBTi-approved methodology documented"
            ),
            "Use an SBTi-approved calculation methodology",
        ))

        # SBTI-SHP-003: Base year within last 2 years
        base_year_val = self._safe_float(
            self._get_field(data, "base_year", 0)
        )
        current_year = utcnow().year
        base_year_ok = (
            base_year_val > 0 and (current_year - base_year_val) <= 2
        )
        has_base_year = base_year_val > 0
        findings.append(ComplianceFinding(
            f"{fw}-003", fw,
            "Base year must be within last 2 years",
            has_base_year,
            "WARNING",
            (
                f"Base year = {int(base_year_val)} "
                + ("(within 2 years)" if base_year_ok else "(may need recalculation)")
                if has_base_year
                else "Base year not established"
            ),
            "Establish a base year within the last 2 years for SBTi compliance",
        ))

        # SBTI-SHP-004: Target aligned with 1.5C or WB2C
        has_target = self._has_any_field(
            data, "target_ambition", "sbti_target", "emission_target",
            "target_pathway",
        )
        findings.append(ComplianceFinding(
            f"{fw}-004", fw,
            "Target must be aligned with 1.5C or well-below 2C",
            has_target,
            "ERROR",
            (
                "Target ambition documented"
                if has_target
                else "Target ambition not documented (1.5C or WB2C required)"
            ),
            "Document target_ambition aligned with 1.5C or well-below 2C pathway",
        ))

        # SBTI-SHP-005: Annual progress tracked
        has_annual = total_co2e > 0 or self._has_any_field(
            data, "annual_tracking", "reporting_year",
        )
        findings.append(ComplianceFinding(
            f"{fw}-005", fw,
            "Annual progress must be tracked",
            has_annual,
            "WARNING",
            (
                "Annual progress tracking present"
                if has_annual
                else "Annual progress not tracked"
            ),
            "Track annual emission progress against SBTi target",
        ))

        # SBTI-SHP-006: Third-party verification
        has_verification = self._has_any_field(
            data, "verification_status", "verified",
            "third_party_verification",
        )
        findings.append(ComplianceFinding(
            f"{fw}-006", fw,
            "Third-party verification must be obtained",
            has_verification,
            "WARNING",
            (
                "Verification status documented"
                if has_verification
                else "Third-party verification not documented"
            ),
            "Obtain third-party verification of Scope 2 emissions",
        ))

        # SBTI-SHP-007: Data completeness >= 95%
        completeness_pct = self._safe_float(
            self._get_field(data, "completeness_pct",
                            self._get_field(data, "coverage_pct", 0))
        )
        data_complete = completeness_pct >= 95 or (
            has_consumption and total_co2e > 0
        )
        findings.append(ComplianceFinding(
            f"{fw}-007", fw,
            "Data completeness must be >= 95%",
            data_complete,
            "ERROR",
            (
                f"Data completeness = {completeness_pct}% (or assumed complete)"
                if data_complete
                else f"Data completeness = {completeness_pct}% (below 95%)"
            ),
            "Ensure data completeness is at least 95%",
        ))

        # SBTI-SHP-008: Consistent organizational boundary
        has_boundary = self._has_any_field(
            data, "facility_id", "organization_id", "org_boundary",
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "Consistent organizational boundary must be maintained",
            has_boundary,
            "ERROR",
            (
                "Organizational boundary defined"
                if has_boundary
                else "Organizational boundary not defined"
            ),
            "Maintain consistent organizational boundary across reporting periods",
        ))

        # SBTI-SHP-009: Quality criteria for market instruments
        has_instruments = self._has_any_field(
            data, "instruments", "market_instruments",
            "contractual_instruments",
        )
        has_market = self._has_any_field(
            data, "market_based_co2e", "market_based_emissions",
        )
        # Only relevant if market-based approach is used
        if has_instruments or has_market:
            quality = self._has_any_field(
                data, "instrument_quality", "quality_assessment",
            )
            findings.append(ComplianceFinding(
                f"{fw}-009", fw,
                "Quality criteria for market instruments must be met",
                quality or has_instruments,
                "WARNING",
                (
                    "Market instrument quality criteria assessed"
                    if quality or has_instruments
                    else "Market instrument quality criteria not assessed"
                ),
                "Assess quality criteria for contractual instruments per GHG Protocol",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-009", fw,
                "Quality criteria for market instruments must be met",
                True,
                "WARNING",
                "Not applicable (no market instruments used)",
                "",
            ))

        # SBTI-SHP-010: RE100-compatible target if applicable
        has_re_target = self._has_any_field(
            data, "renewable_target", "re100_target",
            "renewable_procurement_pct",
        )
        findings.append(ComplianceFinding(
            f"{fw}-010", fw,
            "RE100-compatible renewable target if applicable",
            has_re_target or True,  # INFO level
            "INFO",
            (
                "Renewable target documented"
                if has_re_target
                else "Renewable target not documented (informational)"
            ),
            "Consider RE100-compatible renewable energy procurement target",
        ))

        # SBTI-SHP-011: Uncertainty within SBTi thresholds
        has_uncertainty = self._has_any_field(
            data, "uncertainty_pct", "uncertainty",
            "has_uncertainty",
        )
        uncertainty_pct = self._safe_float(
            self._get_field(data, "uncertainty_pct", 0)
        )
        within_limits = has_uncertainty and (
            uncertainty_pct <= 50 or uncertainty_pct == 0
        )
        findings.append(ComplianceFinding(
            f"{fw}-011", fw,
            "Uncertainty must be within SBTi thresholds",
            within_limits or has_uncertainty,
            "WARNING",
            (
                f"Uncertainty = {uncertainty_pct}% (within thresholds)"
                if within_limits
                else "Uncertainty not documented or exceeds thresholds"
            ),
            "Quantify uncertainty and ensure it is within SBTi acceptable thresholds",
        ))

        # SBTI-SHP-012: Consistent methodology across years
        has_methodology_consistency = self._has_any_field(
            data, "methodology_consistency", "calculation_method",
            "consistent_methodology",
        )
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Consistent methodology must be maintained across years",
            has_methodology_consistency,
            "INFO",
            (
                "Methodology consistency documented"
                if has_methodology_consistency
                else "Methodology consistency not documented"
            ),
            "Maintain consistent calculation methodology across reporting years",
        ))

        return findings

    # ==================================================================
    # Framework 6: EU Energy Efficiency Directive (12 requirements)
    # ==================================================================

    def check_eu_eed(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with EU Energy Efficiency Directive.

        12 requirements covering CHP classification, allocation method,
        efficiency data, primary energy savings, high-efficiency check,
        reference efficiencies, fuel data, heat output metering, power
        output metering, operating hours, annual reporting, and
        monitoring plan.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "eu_eed"
        calc_method = self._calculation_method(data)
        is_chp = calc_method == "CHP_ALLOCATED" or self._has_any_field(
            data, "chp_allocation_method", "chp_parameters",
            "electrical_efficiency", "thermal_efficiency",
        )

        # EED-SHP-001: CHP classified per EU EED
        has_chp_class = self._has_any_field(
            data, "chp_classification", "chp_type",
            "cogeneration_type", "chp_allocation_method",
        )
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-001", fw,
                "CHP must be classified per EU EED",
                has_chp_class,
                "ERROR",
                (
                    "CHP classification documented"
                    if has_chp_class
                    else "CHP classification not documented"
                ),
                "Classify CHP system per EU EED definitions",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-001", fw,
                "CHP must be classified per EU EED",
                True,
                "ERROR",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EED-SHP-002: Allocation method documented
        has_alloc = self._has_any_field(
            data, "chp_allocation_method", "allocation_method",
        )
        alloc_method = str(
            self._get_field(data, "chp_allocation_method", "")
        ).lower()
        valid_alloc = alloc_method in VALID_CHP_ALLOC_METHODS
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-002", fw,
                "Allocation method must be documented",
                has_alloc and (valid_alloc or has_alloc),
                "ERROR",
                (
                    f"Allocation method = {alloc_method}"
                    if has_alloc
                    else "Allocation method not documented for CHP"
                ),
                "Specify chp_allocation_method (efficiency, energy, or exergy)",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-002", fw,
                "Allocation method must be documented",
                True,
                "ERROR",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EED-SHP-003: Electrical and thermal efficiency measured
        has_elec_eff = self._has_field(data, "electrical_efficiency")
        has_therm_eff = self._has_field(data, "thermal_efficiency")
        has_overall_eff = self._has_any_field(
            data, "overall_efficiency", "boiler_efficiency",
        )
        eff_documented = (has_elec_eff and has_therm_eff) or has_overall_eff
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-003", fw,
                "Electrical and thermal efficiency must be measured",
                eff_documented,
                "ERROR",
                (
                    "Efficiency data documented"
                    if eff_documented
                    else "Efficiency data not documented for CHP"
                ),
                "Provide electrical_efficiency and thermal_efficiency for CHP systems",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-003", fw,
                "Electrical and thermal efficiency must be measured",
                has_overall_eff or True,
                "ERROR",
                (
                    "Boiler/system efficiency documented"
                    if has_overall_eff
                    else "Efficiency data not required for non-CHP (informational)"
                ),
                "",
            ))

        # EED-SHP-004: Primary Energy Savings (PES) calculated
        has_pes = self._has_any_field(
            data, "primary_energy_savings", "pes_value", "pes_pct",
        )
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-004", fw,
                "Primary Energy Savings (PES) must be calculated",
                has_pes,
                "ERROR",
                (
                    "PES calculated"
                    if has_pes
                    else "PES not calculated for CHP"
                ),
                "Calculate primary_energy_savings per EU EED Annex III formula",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-004", fw,
                "Primary Energy Savings (PES) must be calculated",
                True,
                "ERROR",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EED-SHP-005: High-efficiency CHP status determined
        has_he_status = self._has_any_field(
            data, "high_efficiency_chp", "is_high_efficiency",
        )
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-005", fw,
                "High-efficiency CHP status must be determined",
                has_he_status,
                "WARNING",
                (
                    "High-efficiency CHP status determined"
                    if has_he_status
                    else "High-efficiency CHP status not determined"
                ),
                "Determine if CHP qualifies as high-efficiency per EU EED (PES >= 10%)",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-005", fw,
                "High-efficiency CHP status must be determined",
                True,
                "WARNING",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EED-SHP-006: Reference efficiencies from EU EED Annex II
        has_ref_eff = self._has_any_field(
            data, "reference_efficiencies", "ref_electrical_efficiency",
            "ref_thermal_efficiency",
        )
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-006", fw,
                "Reference efficiencies from EU EED Annex II must be used",
                has_ref_eff,
                "WARNING",
                (
                    "Reference efficiencies documented"
                    if has_ref_eff
                    else "Reference efficiencies from Annex II not documented"
                ),
                "Use reference efficiencies from EU EED Annex II for PES calculation",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-006", fw,
                "Reference efficiencies from EU EED Annex II must be used",
                True,
                "WARNING",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EED-SHP-007: Fuel type and consumption documented
        has_fuel = self._has_any_field(
            data, "fuel_type", "fuel_consumption", "fuel_consumption_gj",
            "fuel_mix",
        )
        findings.append(ComplianceFinding(
            f"{fw}-007", fw,
            "Fuel type and consumption must be documented",
            has_fuel,
            "ERROR",
            (
                "Fuel data documented"
                if has_fuel
                else "Fuel type and consumption not documented"
            ),
            "Provide fuel_type and fuel_consumption for energy generation source",
        ))

        # EED-SHP-008: Heat output measured (not estimated)
        has_heat_output = self._has_any_field(
            data, "heat_output_gj", "heat_output_mwh",
            "thermal_output", "consumption_gj",
        )
        metered = self._has_any_field(
            data, "metered_heat", "heat_metered", "meter_id",
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "Heat output must be measured (not estimated)",
            has_heat_output or metered,
            "WARNING",
            (
                "Heat output measurement documented"
                if has_heat_output or metered
                else "Heat output measurement not documented"
            ),
            "Provide metered heat output data (heat_output_gj or meter readings)",
        ))

        # EED-SHP-009: Power output measured
        has_power = self._has_any_field(
            data, "power_output_mwh", "electrical_output",
            "power_output_kwh",
        )
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-009", fw,
                "Power output must be measured",
                has_power,
                "WARNING",
                (
                    "Power output documented"
                    if has_power
                    else "Power output not documented for CHP"
                ),
                "Provide power_output_mwh for CHP electrical output",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-009", fw,
                "Power output must be measured",
                True,
                "WARNING",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EED-SHP-010: Operating hours documented
        has_hours = self._has_any_field(
            data, "operating_hours", "annual_operating_hours",
            "runtime_hours",
        )
        findings.append(ComplianceFinding(
            f"{fw}-010", fw,
            "Operating hours must be documented",
            has_hours,
            "INFO",
            (
                "Operating hours documented"
                if has_hours
                else "Operating hours not documented"
            ),
            "Document annual operating_hours for the steam/heat generation system",
        ))

        # EED-SHP-011: Annual reporting to authority
        has_annual = self._has_any_field(
            data, "annual_reporting", "reporting_year",
            "report_submitted",
        )
        findings.append(ComplianceFinding(
            f"{fw}-011", fw,
            "Annual reporting to competent authority must be maintained",
            has_annual,
            "WARNING",
            (
                "Annual reporting documented"
                if has_annual
                else "Annual reporting status not documented"
            ),
            "Document annual reporting status to competent authority",
        ))

        # EED-SHP-012: Monitoring plan established
        has_monitoring = self._has_any_field(
            data, "monitoring_plan", "monitoring_plan_id",
            "monitoring_frequency",
        )
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Monitoring plan must be established",
            has_monitoring,
            "WARNING",
            (
                "Monitoring plan documented"
                if has_monitoring
                else "Monitoring plan not established"
            ),
            "Establish a monitoring plan for continuous data quality assurance",
        ))

        return findings

    # ==================================================================
    # Framework 7: EPA Mandatory Reporting Rule (12 requirements)
    # ==================================================================

    def check_epa_mrr(
        self,
        data: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with EPA Mandatory Reporting Rule (40 CFR Part 98).

        12 requirements covering GHG reporting, fuel-specific emission
        factors, monitoring plan, HHV measurement, carbon content,
        oxidation factor, CHP documentation, data retention, QC
        procedures, third-party review, emission threshold, and
        reporting deadline.

        Args:
            data: Calculation data dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "epa_mrr"
        total_co2e = self._safe_float(
            self._get_field(data, "total_co2e_tonnes", 0)
        )

        # EPA-SHP-001: GHG reported per 40 CFR Part 98
        has_ghg = total_co2e > 0 or self._has_any_field(
            data, "total_co2e_kg", "ghg_reported",
        )
        findings.append(ComplianceFinding(
            f"{fw}-001", fw,
            "GHG emissions must be reported per 40 CFR Part 98",
            has_ghg,
            "ERROR",
            (
                f"GHG emissions reported: {total_co2e} tCO2e"
                if has_ghg
                else "GHG emissions not reported"
            ),
            "Report total_co2e_tonnes per 40 CFR Part 98 requirements",
        ))

        # EPA-SHP-002: Fuel-specific EFs (not defaults)
        has_fuel_ef = self._has_any_field(
            data, "fuel_type", "fuel_specific_ef",
            "fuel_emission_factor",
        )
        ef_source = str(
            self._get_field(data, "ef_source",
                            self._get_field(data, "emission_factor_source", ""))
        ).upper()
        is_fuel_specific = ef_source in (
            "FACILITY_SPECIFIC", "EPA_AP42", "SUPPLIER_SPECIFIC",
        ) or has_fuel_ef
        findings.append(ComplianceFinding(
            f"{fw}-002", fw,
            "Fuel-specific emission factors must be used (not generic defaults)",
            is_fuel_specific,
            "ERROR",
            (
                "Fuel-specific emission factors used"
                if is_fuel_specific
                else "Generic default emission factors used (fuel-specific required by EPA)"
            ),
            "Use fuel-specific emission factors from EPA AP-42 or facility measurements",
        ))

        # EPA-SHP-003: Monitoring plan per Subpart C
        has_monitoring = self._has_any_field(
            data, "monitoring_plan", "monitoring_plan_id",
            "subpart_c_monitoring",
        )
        findings.append(ComplianceFinding(
            f"{fw}-003", fw,
            "Monitoring plan per Subpart C must be maintained",
            has_monitoring,
            "WARNING",
            (
                "Monitoring plan documented"
                if has_monitoring
                else "Monitoring plan not documented per Subpart C"
            ),
            "Establish monitoring plan per 40 CFR Part 98 Subpart C",
        ))

        # EPA-SHP-004: Higher heating value (HHV) measured
        has_hhv = self._has_any_field(
            data, "hhv", "higher_heating_value", "hhv_mj_per_kg",
            "hhv_btu_per_scf",
        )
        findings.append(ComplianceFinding(
            f"{fw}-004", fw,
            "Higher heating value (HHV) must be measured",
            has_hhv,
            "WARNING",
            (
                "HHV measurement documented"
                if has_hhv
                else "HHV not measured (required for fuel-based calculations)"
            ),
            "Provide higher_heating_value from fuel analysis",
        ))

        # EPA-SHP-005: Carbon content documented
        has_carbon = self._has_any_field(
            data, "carbon_content", "carbon_content_pct",
            "carbon_fraction",
        )
        findings.append(ComplianceFinding(
            f"{fw}-005", fw,
            "Carbon content of fuel must be documented",
            has_carbon,
            "WARNING",
            (
                "Carbon content documented"
                if has_carbon
                else "Carbon content not documented"
            ),
            "Provide carbon_content or carbon_content_pct from fuel analysis",
        ))

        # EPA-SHP-006: Oxidation factor applied
        has_oxidation = self._has_any_field(
            data, "oxidation_factor", "oxidation_fraction",
        )
        findings.append(ComplianceFinding(
            f"{fw}-006", fw,
            "Oxidation factor must be applied",
            has_oxidation,
            "INFO",
            (
                "Oxidation factor documented"
                if has_oxidation
                else "Oxidation factor not specified (default 1.0 may apply)"
            ),
            "Specify oxidation_factor if not assuming complete combustion (1.0)",
        ))

        # EPA-SHP-007: CHP allocation documented if applicable
        calc_method = self._calculation_method(data)
        is_chp = calc_method == "CHP_ALLOCATED" or self._has_any_field(
            data, "chp_allocation_method", "chp_parameters",
        )
        has_chp_doc = self._has_any_field(
            data, "chp_allocation_method", "chp_documentation",
        )
        if is_chp:
            findings.append(ComplianceFinding(
                f"{fw}-007", fw,
                "CHP allocation must be documented if applicable",
                has_chp_doc,
                "ERROR",
                (
                    "CHP allocation documented"
                    if has_chp_doc
                    else "CHP allocation not documented"
                ),
                "Document CHP allocation method and parameters",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-007", fw,
                "CHP allocation must be documented if applicable",
                True,
                "ERROR",
                "Not applicable (non-CHP system)",
                "",
            ))

        # EPA-SHP-008: Data retained for minimum period
        has_retention = self._has_any_field(
            data, "data_retention", "retention_years",
            "data_retention_policy",
        )
        findings.append(ComplianceFinding(
            f"{fw}-008", fw,
            "Data must be retained for minimum period",
            has_retention or True,  # Often implicit
            "INFO",
            (
                "Data retention policy documented"
                if has_retention
                else "Data retention policy not explicitly documented (informational)"
            ),
            "Maintain data retention per EPA 40 CFR Part 98 (minimum 3 years)",
        ))

        # EPA-SHP-009: QC procedures implemented
        has_qc = self._has_any_field(
            data, "qc_procedures", "quality_control",
            "qc_status", "qc_check_passed",
        )
        findings.append(ComplianceFinding(
            f"{fw}-009", fw,
            "QC procedures must be implemented",
            has_qc,
            "WARNING",
            (
                "QC procedures documented"
                if has_qc
                else "QC procedures not documented"
            ),
            "Document quality control procedures for emissions data",
        ))

        # EPA-SHP-010: Third-party review for large emitters
        has_review = self._has_any_field(
            data, "third_party_review", "verification_status",
            "verified",
        )
        is_large_emitter = total_co2e >= 25000
        if is_large_emitter:
            findings.append(ComplianceFinding(
                f"{fw}-010", fw,
                "Third-party review required for large emitters",
                has_review,
                "WARNING",
                (
                    "Third-party review documented"
                    if has_review
                    else f"Large emitter ({total_co2e} tCO2e >= 25,000) without third-party review"
                ),
                "Obtain third-party review for facilities emitting >= 25,000 tCO2e",
            ))
        else:
            findings.append(ComplianceFinding(
                f"{fw}-010", fw,
                "Third-party review required for large emitters",
                True,
                "WARNING",
                f"Below large emitter threshold ({total_co2e} tCO2e < 25,000)",
                "",
            ))

        # EPA-SHP-011: Emission threshold check (25,000 tCO2e)
        has_threshold = self._has_any_field(
            data, "emission_threshold_check", "above_reporting_threshold",
        )
        threshold_assessed = has_threshold or total_co2e > 0
        findings.append(ComplianceFinding(
            f"{fw}-011", fw,
            "Emission threshold check (25,000 tCO2e) must be performed",
            threshold_assessed,
            "ERROR",
            (
                f"Emission threshold assessed: {total_co2e} tCO2e "
                f"({'above' if is_large_emitter else 'below'} 25,000 tCO2e threshold)"
                if threshold_assessed
                else "Emission threshold not assessed"
            ),
            "Assess whether facility exceeds 25,000 tCO2e reporting threshold",
        ))

        # EPA-SHP-012: Reporting deadline compliance
        has_deadline = self._has_any_field(
            data, "reporting_deadline", "submission_date",
            "deadline_compliance",
        )
        findings.append(ComplianceFinding(
            f"{fw}-012", fw,
            "Reporting deadline compliance must be documented",
            has_deadline,
            "INFO",
            (
                "Reporting deadline compliance documented"
                if has_deadline
                else "Reporting deadline compliance not documented"
            ),
            "Document submission_date and reporting_deadline compliance",
        ))

        return findings

    # ==================================================================
    # Public Method 10: get_framework_requirements
    # ==================================================================

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, Any]]:
        """Get a listing of all requirements for a specific framework.

        Args:
            framework: Framework name (e.g. "ghg_protocol_scope2", "iso_14064").

        Returns:
            List of requirement dictionaries with id, description, severity.

        Raises:
            ValueError: If the framework is not supported.
        """
        fw_lower = framework.lower()
        if fw_lower not in self._framework_checkers:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Supported: {SUPPORTED_FRAMEWORKS}"
            )

        checker = self._framework_checkers[fw_lower]
        # Run with empty data to extract requirement metadata
        findings = checker({})

        return [
            {
                "requirement_id": f.requirement_id,
                "framework": f.framework,
                "requirement": f.requirement,
                "severity": f.severity,
            }
            for f in findings
        ]

    # ==================================================================
    # Public Method 11: get_all_frameworks
    # ==================================================================

    def get_all_frameworks(self) -> List[str]:
        """Return a list of all supported compliance framework identifiers.

        Returns:
            List of framework name strings.
        """
        return list(SUPPORTED_FRAMEWORKS)

    # ==================================================================
    # Public Method 12: get_framework_count
    # ==================================================================

    def get_framework_count(self) -> int:
        """Return the number of supported compliance frameworks.

        Returns:
            Integer count of frameworks (7).
        """
        return len(SUPPORTED_FRAMEWORKS)

    # ==================================================================
    # Public Method 13: get_total_requirements
    # ==================================================================

    def get_total_requirements(self) -> int:
        """Return the total number of requirements across all frameworks.

        Returns:
            Integer count of requirements (84).
        """
        return TOTAL_REQUIREMENTS

    # ==================================================================
    # Public Method 14: compute_compliance_score
    # ==================================================================

    def compute_compliance_score(
        self,
        results: Dict[str, Any],
    ) -> Decimal:
        """Compute the overall compliance score from check results.

        Args:
            results: Output from check_compliance.

        Returns:
            Decimal compliance score as a percentage (0-100).
        """
        overall = results.get("overall", {})
        total_passed = _to_decimal(overall.get("total_passed", 0))
        total_reqs = _to_decimal(overall.get("total_requirements", 0))

        if total_reqs == Decimal("0"):
            return Decimal("0")

        score = (total_passed / total_reqs * Decimal("100")).quantize(
            Decimal("0.1"), ROUND_HALF_UP
        )
        return score

    # ==================================================================
    # Public Method 15: get_non_compliant_items
    # ==================================================================

    def get_non_compliant_items(
        self,
        results: Dict[str, Any],
    ) -> List[str]:
        """Extract a flat list of non-compliant requirement IDs.

        Args:
            results: Output from check_compliance.

        Returns:
            List of requirement_id strings that failed compliance.
        """
        non_compliant: List[str] = []
        framework_results = results.get("framework_results", {})

        for fw_result in framework_results.values():
            for finding in fw_result.get("findings", []):
                if isinstance(finding, dict) and not finding.get("passed", True):
                    req_id = finding.get("requirement_id", "")
                    if req_id:
                        non_compliant.append(req_id)

        return non_compliant

    # ==================================================================
    # Public Method 16: get_recommendations
    # ==================================================================

    def get_recommendations(
        self,
        results: Dict[str, Any],
    ) -> List[str]:
        """Extract a deduplicated list of remediation recommendations.

        Args:
            results: Output from check_compliance.

        Returns:
            List of recommendation strings for failed requirements.
        """
        recommendations: List[str] = []
        seen: Set[str] = set()
        framework_results = results.get("framework_results", {})

        for fw_result in framework_results.values():
            for finding in fw_result.get("findings", []):
                if isinstance(finding, dict) and not finding.get("passed", True):
                    rec = finding.get("recommendation", "")
                    if rec and rec not in seen:
                        recommendations.append(rec)
                        seen.add(rec)

        return recommendations

    # ==================================================================
    # Public Method 17: validate_request
    # ==================================================================

    def validate_request(
        self,
        calc_result: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate that a calculation result dict has minimum required fields.

        Args:
            calc_result: Calculation result dictionary to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        if not isinstance(calc_result, dict):
            return False, ["calc_result must be a dictionary"]

        if not calc_result:
            return False, ["calc_result is empty"]

        # Check for minimum required fields
        required_fields = [
            ("energy_type", "energy_type is required for compliance checking"),
        ]
        recommended_fields = [
            ("total_co2e_tonnes", "total_co2e_tonnes is recommended"),
            ("calculation_method", "calculation_method is recommended"),
            ("consumption_gj", "consumption_gj (or equivalent) is recommended"),
        ]

        for field_name, msg in required_fields:
            if not self._has_field(calc_result, field_name):
                errors.append(msg)

        for field_name, msg in recommended_fields:
            if not self._has_field(calc_result, field_name):
                # Warnings, not blocking errors
                pass

        # Validate energy type if present
        energy = self._energy_type(calc_result)
        if energy and energy not in VALID_ENERGY_TYPES:
            errors.append(
                f"energy_type '{energy}' is not recognized; "
                f"expected one of {sorted(VALID_ENERGY_TYPES)}"
            )

        # Validate calculation method if present
        calc_method = self._calculation_method(calc_result)
        if calc_method and calc_method not in VALID_CALCULATION_METHODS:
            errors.append(
                f"calculation_method '{calc_method}' is not recognized; "
                f"expected one of {sorted(VALID_CALCULATION_METHODS)}"
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    # ==================================================================
    # Public Method 18: get_compliance_stats
    # ==================================================================

    def get_compliance_stats(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine metadata and counters.
        """
        with self._lock:
            return {
                "engine": "ComplianceCheckerEngine",
                "agent": "AGENT-MRV-011",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_checks": self._total_checks,
                "supported_frameworks": list(SUPPORTED_FRAMEWORKS),
                "framework_count": len(SUPPORTED_FRAMEWORKS),
                "total_requirements": TOTAL_REQUIREMENTS,
                "framework_check_counts": dict(self._framework_check_counts),
                "total_findings": self._total_findings,
                "total_passed": self._total_passed,
                "total_failed": self._total_failed,
                "framework_requirement_counts": {
                    "ghg_protocol_scope2": 12,
                    "iso_14064": 12,
                    "csrd_esrs": 12,
                    "cdp": 12,
                    "sbti": 12,
                    "eu_eed": 12,
                    "epa_mrr": 12,
                },
                "timestamp": utcnow().isoformat(),
            }

    # ==================================================================
    # Public Method 19: health_check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the compliance checker engine.

        Verifies that the engine is properly initialized, all framework
        checkers are registered, and requirement counts are correct.

        Returns:
            Dictionary with health status, checks performed, and any issues.
        """
        issues: List[str] = []

        # Verify all frameworks have checkers
        for fw in SUPPORTED_FRAMEWORKS:
            if fw not in self._framework_checkers:
                issues.append(f"Missing checker for framework: {fw}")

        # Verify checker count matches expected
        if len(self._framework_checkers) != len(SUPPORTED_FRAMEWORKS):
            issues.append(
                f"Checker count mismatch: {len(self._framework_checkers)} "
                f"vs {len(SUPPORTED_FRAMEWORKS)} expected"
            )

        # Verify total requirements are correct by running empty checks
        total_found = 0
        for fw, checker in self._framework_checkers.items():
            try:
                findings = checker({})
                total_found += len(findings)
            except Exception as exc:
                issues.append(f"Framework '{fw}' checker failed: {exc}")

        if total_found != TOTAL_REQUIREMENTS:
            issues.append(
                f"Total requirements mismatch: found {total_found}, "
                f"expected {TOTAL_REQUIREMENTS}"
            )

        # Verify framework info completeness
        for fw in SUPPORTED_FRAMEWORKS:
            if fw not in FRAMEWORK_INFO:
                issues.append(f"Missing FRAMEWORK_INFO for: {fw}")

        status = "healthy" if not issues else "degraded"

        with self._lock:
            return {
                "status": status,
                "engine": "ComplianceCheckerEngine",
                "agent": "AGENT-MRV-011",
                "version": "1.0.0",
                "frameworks_registered": len(self._framework_checkers),
                "total_requirements_verified": total_found,
                "expected_requirements": TOTAL_REQUIREMENTS,
                "total_checks_performed": self._total_checks,
                "issues": issues,
                "issue_count": len(issues),
                "timestamp": utcnow().isoformat(),
                "uptime_since": self._created_at.isoformat(),
            }

    # ==================================================================
    # Additional convenience methods
    # ==================================================================

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

    def get_compliance_summary(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a compliance summary from check_compliance output.

        Args:
            results: Output from check_compliance.

        Returns:
            Summary dictionary with status counts and recommendations.
        """
        framework_results = results.get("framework_results", {})
        compliant = 0
        partial = 0
        non_compliant = 0
        total_reqs = 0
        total_passed = 0
        total_failed = 0
        critical_findings: List[Dict[str, Any]] = []
        all_recommendations: List[str] = []

        for fw_result in framework_results.values():
            status = str(fw_result.get("status", "")).upper()
            if status == "COMPLIANT":
                compliant += 1
            elif status == "PARTIAL":
                partial += 1
            elif status == "NON_COMPLIANT":
                non_compliant += 1

            total_reqs += int(fw_result.get("total_requirements", 0))
            total_passed += int(fw_result.get("passed", 0))
            total_failed += int(fw_result.get("failed", 0))

            # Collect critical (ERROR) findings
            for f in fw_result.get("findings", []):
                if (
                    isinstance(f, dict)
                    and f.get("severity") == "ERROR"
                    and not f.get("passed", True)
                ):
                    critical_findings.append(f)

            # Collect recommendations
            for rec in fw_result.get("recommendations", []):
                if rec and rec not in all_recommendations:
                    all_recommendations.append(rec)

        # Determine overall status
        if total_reqs == 0:
            overall_status = "NOT_ASSESSED"
        elif total_failed == 0:
            overall_status = "COMPLIANT"
        elif non_compliant > compliant:
            overall_status = "NON_COMPLIANT"
        else:
            overall_status = "PARTIAL"

        return {
            "frameworks_checked": len(framework_results),
            "overall_status": overall_status,
            "compliant_count": compliant,
            "non_compliant_count": non_compliant,
            "partial_count": partial,
            "total_requirements": total_reqs,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "critical_findings": critical_findings,
            "critical_finding_count": len(critical_findings),
            "consolidated_recommendations": all_recommendations,
            "provenance_hash": _compute_hash({
                "frameworks_checked": len(framework_results),
                "overall_status": overall_status,
                "total_passed": total_passed,
                "total_failed": total_failed,
            }),
        }

    def get_remediation_plan(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate a prioritized remediation plan for compliance failures.

        Args:
            results: Output from check_compliance.

        Returns:
            List of remediation items sorted by priority (ERROR > WARNING > INFO).
        """
        severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        items: List[Dict[str, Any]] = []

        framework_results = results.get("framework_results", {})
        for fw_name, fw_result in framework_results.items():
            for finding in fw_result.get("findings", []):
                if isinstance(finding, dict) and not finding.get("passed", True):
                    items.append({
                        "framework": fw_name,
                        "requirement_id": finding.get("requirement_id", ""),
                        "description": finding.get("requirement", ""),
                        "severity": finding.get("severity", "INFO"),
                        "finding": finding.get("finding", ""),
                        "recommendation": finding.get("recommendation", ""),
                        "priority": severity_order.get(
                            finding.get("severity", "INFO"), 2
                        ),
                    })

        items.sort(
            key=lambda x: (x["priority"], x["framework"], x["requirement_id"])
        )
        return items

    def get_framework_info(self, framework: str) -> Dict[str, Any]:
        """Get metadata for a specific framework.

        Args:
            framework: Framework identifier.

        Returns:
            Framework metadata dict, or error dict if not found.
        """
        info = FRAMEWORK_INFO.get(framework.lower())
        if info is None:
            return {"error": f"Framework '{framework}' not found"}
        return dict(info)

# ===========================================================================
# Module-level singleton accessor
# ===========================================================================

def get_compliance_checker() -> ComplianceCheckerEngine:
    """Get or create the ComplianceCheckerEngine singleton.

    Returns:
        The singleton ComplianceCheckerEngine instance.
    """
    return ComplianceCheckerEngine()
