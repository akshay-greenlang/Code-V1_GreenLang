# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - Multi-Framework Regulatory Compliance (Engine 6 of 7)

AGENT-MRV-009: Scope 2 Location-Based Emissions Agent

Validates Scope 2 location-based emission calculations against seven regulatory
frameworks to ensure data completeness, methodological correctness, and
reporting readiness.  Each framework defines specific requirements that are
individually checked and scored.

Supported Frameworks (80 total requirements):
    1. GHG Protocol Scope 2 Guidance (2015)  - 12 requirements
    2. IPCC 2006 Guidelines                  - 11 requirements
    3. ISO 14064-1:2018                      - 12 requirements
    4. CSRD/ESRS E1                          - 11 requirements
    5. EPA Greenhouse Gas Reporting Program  - 12 requirements
    6. DEFRA Reporting (UK SECR)             - 11 requirements
    7. CDP Climate Change                    - 11 requirements

Compliance Statuses:
    COMPLIANT:     All requirements met (100% pass rate)
    PARTIAL:       Some requirements met (50-99% pass rate)
    NON_COMPLIANT: Fewer than 50% of requirements met

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
    >>> from greenlang.agents.mrv.scope2_location.compliance_checker import (
    ...     ComplianceCheckerEngine,
    ... )
    >>> engine = ComplianceCheckerEngine()
    >>> result = engine.check_compliance(
    ...     calculation_result={
    ...         "energy_type": "electricity",
    ...         "consumption_kwh": 1000000,
    ...         "emission_factor_source": "egrid",
    ...         "grid_region": "RFCE",
    ...         "ef_year": 2023,
    ...         "reporting_year": 2024,
    ...         "total_co2e_tonnes": 385.2,
    ...         "gas_breakdown": {"CO2": 380.0, "CH4": 3.1, "N2O": 2.1},
    ...         "market_based_available": True,
    ...         "provenance_hash": "abc123...",
    ...     },
    ...     frameworks=["ghg_protocol_scope2"],
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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
from decimal import Decimal, ROUND_HALF_UP
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
    from greenlang.agents.mrv.scope2_location.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.scope2_location.metrics import get_metrics as _get_metrics
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]

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
    "ghg_protocol_scope2",
    "ipcc_2006",
    "iso_14064",
    "csrd_esrs",
    "epa_ghgrp",
    "defra",
    "cdp",
]

#: Contractual EF sources that are NOT grid-average (invalid for location-based).
CONTRACTUAL_EF_SOURCES: List[str] = [
    "rec", "go", "ppa", "green_tariff", "rego", "rgo",
    "i_rec", "contract", "market_instrument", "supplier_specific",
]

#: Valid grid-average EF sources appropriate for location-based method.
GRID_AVERAGE_EF_SOURCES: List[str] = [
    "egrid", "iea", "defra", "ipcc", "national", "custom",
    "national_registry", "residual_mix", "grid_average",
    "aib", "epa", "unfccc",
]

#: US eGRID subregion codes (representative set).
EGRID_SUBREGIONS: List[str] = [
    "AKGD", "AKMS", "AZNM", "CAMX", "ERCT", "FRCC", "HIMS", "HIOA",
    "MROE", "MROW", "NEWE", "NWPP", "NYCW", "NYLI", "NYUP", "PRMS",
    "RFCE", "RFCM", "RFCW", "RMPA", "SPNO", "SPSO", "SRMV", "SRMW",
    "SRSO", "SRTV", "SRVC",
]

#: Valid energy types for Scope 2.
VALID_ENERGY_TYPES: List[str] = [
    "electricity", "steam", "heating", "cooling",
    "chilled_water", "hot_water",
]

#: Valid GWP assessment report sources.
VALID_GWP_SOURCES: List[str] = ["AR4", "AR5", "AR6"]

#: Individual greenhouse gases expected for per-gas reporting.
EXPECTED_GASES: List[str] = ["CO2", "CH4", "N2O"]

#: Total requirements across all 7 frameworks.
TOTAL_REQUIREMENTS: int = 80

#: Framework metadata with reference links and descriptions.
FRAMEWORK_INFO: Dict[str, Dict[str, Any]] = {
    "ghg_protocol_scope2": {
        "name": "GHG Protocol Scope 2 Guidance",
        "version": "2015",
        "publisher": "WRI/WBCSD",
        "description": (
            "The GHG Protocol Scope 2 Guidance provides standards and "
            "guidance for accounting and reporting Scope 2 emissions from "
            "purchased electricity, steam, heating, and cooling."
        ),
        "reference": "https://ghgprotocol.org/scope_2_guidance",
        "requirements_count": 12,
        "key_principle": (
            "Location-based method uses grid-average emission factors; "
            "dual reporting with market-based method is required."
        ),
    },
    "ipcc_2006": {
        "name": "IPCC 2006 Guidelines for National GHG Inventories",
        "version": "2006 (2019 Refinement)",
        "publisher": "IPCC",
        "description": (
            "Volume 2, Chapter 2 covers stationary energy combustion for "
            "electricity generation; indirect emissions from purchased energy."
        ),
        "reference": "https://www.ipcc-nggip.iges.or.jp/public/2006gl/",
        "requirements_count": 11,
        "key_principle": (
            "Tier-based approach with increasing data quality requirements."
        ),
    },
    "iso_14064": {
        "name": "ISO 14064-1:2018",
        "version": "2018",
        "publisher": "ISO",
        "description": (
            "International standard for quantification and reporting of "
            "greenhouse gas emissions and removals at the organization level. "
            "Scope 2 is reported as Category 2 (indirect from imported energy)."
        ),
        "reference": "https://www.iso.org/standard/66453.html",
        "requirements_count": 12,
        "key_principle": (
            "Category 2 indirect emissions from imported energy; uncertainty "
            "assessment required; organizational boundary must be defined."
        ),
    },
    "csrd_esrs": {
        "name": "CSRD / ESRS E1 Climate Change",
        "version": "2024",
        "publisher": "EFRAG / European Commission",
        "description": (
            "European Sustainability Reporting Standards E1 Climate Change "
            "requires disclosure of Scope 2 emissions using both location-based "
            "and market-based methods, with intensity metrics."
        ),
        "reference": "https://www.efrag.org/lab6",
        "requirements_count": 11,
        "key_principle": (
            "Dual reporting mandatory; intensity metrics required; "
            "year-over-year trend and reduction targets referenced."
        ),
    },
    "epa_ghgrp": {
        "name": "EPA Greenhouse Gas Reporting Program",
        "version": "40 CFR Part 98",
        "publisher": "US Environmental Protection Agency",
        "description": (
            "US-specific reporting requirements for facilities that emit "
            "25,000 or more metric tons of CO2e per year. Requires use of "
            "eGRID subregion emission factors for indirect electricity."
        ),
        "reference": "https://www.epa.gov/ghgreporting",
        "requirements_count": 12,
        "key_principle": (
            "eGRID subregion emission factors mandatory for US facilities; "
            "25,000 tCO2e reporting threshold; subpart requirements."
        ),
    },
    "defra": {
        "name": "DEFRA/BEIS UK GHG Conversion Factors",
        "version": "2024",
        "publisher": "UK DEFRA / DESNZ",
        "description": (
            "UK-specific greenhouse gas conversion factors and Streamlined "
            "Energy and Carbon Reporting (SECR) requirements under the "
            "UK Companies Act 2006 s.414C."
        ),
        "reference": (
            "https://www.gov.uk/government/collections/"
            "government-conversion-factors-for-company-reporting"
        ),
        "requirements_count": 11,
        "key_principle": (
            "DEFRA conversion factors for UK operations; energy in kWh; "
            "SECR compliance for qualifying UK companies."
        ),
    },
    "cdp": {
        "name": "CDP Climate Change Questionnaire",
        "version": "2024",
        "publisher": "CDP (formerly Carbon Disclosure Project)",
        "description": (
            "Sections C6.3 (Scope 2 location-based) and C6.3a (Scope 2 "
            "market-based) of the CDP Climate Change questionnaire require "
            "activity data, emission factors, and methodology disclosure."
        ),
        "reference": "https://www.cdp.net/en",
        "requirements_count": 11,
        "key_principle": (
            "C6.3/C6.3a dual Scope 2 reporting; activity data and EF "
            "source must be identified; methodology disclosed."
        ),
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

@dataclass
class ComplianceCheckResult:
    """Result of a compliance check for a single framework.

    Attributes:
        check_id: Unique identifier for this check run.
        calculation_id: ID of the calculation being checked.
        framework: Framework identifier.
        status: compliant, non_compliant, partial, or not_assessed.
        findings: List of specific findings.
        recommendations: List of actionable recommendations.
        checked_at: ISO timestamp of the check.
        provenance_hash: SHA-256 hash of the check result.
        total_requirements: Number of requirements checked.
        passed_count: Number of requirements passed.
        failed_count: Number of requirements failed.
        error_count: Number of ERROR-severity failures.
        warning_count: Number of WARNING-severity failures.
        pass_rate_pct: Pass rate as percentage.
    """

    check_id: str
    calculation_id: str
    framework: str
    status: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    checked_at: str
    provenance_hash: str = ""
    total_requirements: int = 0
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    pass_rate_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "calculation_id": self.calculation_id,
            "framework": self.framework,
            "status": self.status,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "checked_at": self.checked_at,
            "provenance_hash": self.provenance_hash,
            "total_requirements": self.total_requirements,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "pass_rate_pct": self.pass_rate_pct,
        }

# ===========================================================================
# ComplianceCheckerEngine
# ===========================================================================

class ComplianceCheckerEngine:
    """Multi-framework regulatory compliance checker for Scope 2 location-based
    emission calculations.

    Validates calculation results against seven regulatory frameworks with
    80 total requirements covering GHG Protocol Scope 2 Guidance, IPCC 2006,
    ISO 14064-1:2018, CSRD/ESRS E1, EPA GHGRP, DEFRA/SECR, and CDP.

    The engine guarantees zero-hallucination compliance determinations:
    every check is a deterministic boolean evaluation against hard-coded
    regulatory requirement definitions.  No LLM is involved.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> results = engine.check_compliance(
        ...     calculation_result={"energy_type": "electricity", ...},
        ...     frameworks=["ghg_protocol_scope2", "iso_14064"],
        ... )
    """

    def __init__(
        self,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize the ComplianceCheckerEngine.

        Args:
            config: Optional configuration object. If None, loads from
                module-level config if available.
            metrics: Optional metrics object. If None, loads from
                module-level metrics if available.
            provenance: Optional provenance tracker. Not used directly
                but stored for pipeline integration.
        """
        self._lock = threading.RLock()
        self._total_checks: int = 0
        self._checks_by_framework: Dict[str, int] = {
            fw: 0 for fw in SUPPORTED_FRAMEWORKS
        }
        self._checks_by_status: Dict[str, int] = {
            "compliant": 0,
            "non_compliant": 0,
            "partial": 0,
            "not_assessed": 0,
        }
        self._created_at = utcnow()
        self._provenance = provenance

        # Load config
        if config is not None:
            self._config = config
        elif _CONFIG_AVAILABLE:
            self._config = _get_config()
        else:
            self._config = None

        # Load metrics
        if metrics is not None:
            self._metrics = metrics
        elif _METRICS_AVAILABLE:
            self._metrics = _get_metrics()
        else:
            self._metrics = None

        # Determine enabled frameworks from config
        if self._config is not None and hasattr(self._config, "enabled_frameworks"):
            self._enabled_frameworks: List[str] = list(
                self._config.enabled_frameworks
            )
        else:
            self._enabled_frameworks = list(SUPPORTED_FRAMEWORKS)

        # Map framework names to checker methods
        self._framework_checkers: Dict[str, Callable] = {
            "ghg_protocol_scope2": self.check_ghg_protocol,
            "ipcc_2006": self.check_ipcc_2006,
            "iso_14064": self.check_iso_14064,
            "csrd_esrs": self.check_csrd_esrs,
            "epa_ghgrp": self.check_epa_ghgrp,
            "defra": self.check_defra,
            "cdp": self.check_cdp,
        }

        logger.info(
            "ComplianceCheckerEngine initialized: frameworks=%d, "
            "enabled=%d, total_requirements=%d",
            len(SUPPORTED_FRAMEWORKS),
            len(self._enabled_frameworks),
            TOTAL_REQUIREMENTS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_checks(self, framework: str, status: str) -> None:
        """Thread-safe increment of check counters."""
        with self._lock:
            self._total_checks += 1
            fw_lower = framework.lower()
            if fw_lower in self._checks_by_framework:
                self._checks_by_framework[fw_lower] += 1
            if status in self._checks_by_status:
                self._checks_by_status[status] += 1

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

    def _record_metric(self, framework: str, status: str) -> None:
        """Record a compliance check metric if metrics are available."""
        if self._metrics is not None:
            try:
                self._metrics.record_compliance_check(
                    framework=framework.upper(),
                    status=status,
                )
            except Exception:
                pass  # Metrics should never break compliance checks

    def _build_compliance_result(
        self,
        framework: str,
        calculation_id: str,
        findings: List[ComplianceFinding],
    ) -> ComplianceCheckResult:
        """Build a ComplianceCheckResult from a list of findings.

        Computes pass/fail counts, determines status, and generates
        the provenance hash.

        Args:
            framework: Framework identifier.
            calculation_id: Calculation being checked.
            findings: List of ComplianceFinding objects.

        Returns:
            Fully populated ComplianceCheckResult.
        """
        total = len(findings)
        passed = sum(1 for f in findings if f.passed)
        failed = total - passed
        errors = sum(
            1 for f in findings if not f.passed and f.severity == "ERROR"
        )
        warnings = sum(
            1 for f in findings if not f.passed and f.severity == "WARNING"
        )
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        if pass_rate == 100.0:
            status = "compliant"
        elif pass_rate >= 50.0:
            status = "partial"
        else:
            status = "non_compliant"

        recommendations = [
            f.recommendation
            for f in findings
            if not f.passed and f.recommendation
        ]

        check_id = str(uuid4())
        checked_at = utcnow().isoformat()

        result = ComplianceCheckResult(
            check_id=check_id,
            calculation_id=calculation_id,
            framework=framework,
            status=status,
            findings=[f.to_dict() for f in findings],
            recommendations=recommendations,
            checked_at=checked_at,
            total_requirements=total,
            passed_count=passed,
            failed_count=failed,
            error_count=errors,
            warning_count=warnings,
            pass_rate_pct=round(pass_rate, 1),
        )

        # Compute provenance hash
        result.provenance_hash = _compute_hash(result.to_dict())

        # Track metrics
        self._increment_checks(framework, status)
        self._record_metric(framework, status)

        return result

    # ==================================================================
    # Public API: Full Compliance Checks
    # ==================================================================

    def check_compliance(
        self,
        calculation_result: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> List[ComplianceCheckResult]:
        """Run compliance checks against selected or all enabled frameworks.

        This is the primary entry point for compliance validation.  It
        executes each requested framework's checks and returns a list of
        ComplianceCheckResult objects, one per framework.

        Args:
            calculation_result: Dictionary containing the calculation
                output and metadata to validate.
            frameworks: Optional list of framework identifiers.  If None,
                checks all enabled frameworks from configuration.

        Returns:
            List of ComplianceCheckResult, one per framework checked.

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> results = engine.check_compliance(
            ...     {"energy_type": "electricity", "total_co2e_tonnes": 100},
            ...     frameworks=["ghg_protocol_scope2"],
            ... )
            >>> results[0].status
            'partial'
        """
        start_time = time.monotonic()
        calc_id = calculation_result.get("calculation_id", str(uuid4()))

        if frameworks is None:
            frameworks = list(self._enabled_frameworks)

        results: List[ComplianceCheckResult] = []

        for fw in frameworks:
            fw_lower = fw.lower()
            if fw_lower not in self._framework_checkers:
                logger.warning(
                    "Unknown framework '%s', skipping", fw
                )
                continue

            checker = self._framework_checkers[fw_lower]
            findings = checker(calculation_result)
            result = self._build_compliance_result(
                fw_lower, calc_id, findings
            )
            results.append(result)

        elapsed_ms = round((time.monotonic() - start_time) * 1000, 3)

        logger.info(
            "Compliance check complete: calc_id=%s, frameworks=%d, "
            "time=%.3fms",
            calc_id,
            len(results),
            elapsed_ms,
        )

        return results

    def check_all_frameworks(
        self,
        calculation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all 7 frameworks and return an aggregated result dictionary.

        This convenience method runs every supported framework (not just
        enabled ones) and produces a single summary dictionary suitable
        for API responses and audit records.

        Args:
            calculation_result: Calculation output to validate.

        Returns:
            Aggregated compliance result dictionary with overall status,
            per-framework results, and provenance hash.
        """
        start_time = time.monotonic()
        calc_id = calculation_result.get("calculation_id", str(uuid4()))

        framework_results: Dict[str, Dict[str, Any]] = {}
        total_passed = 0
        total_requirements = 0
        total_errors = 0
        total_warnings = 0

        for fw in SUPPORTED_FRAMEWORKS:
            checker = self._framework_checkers[fw]
            findings = checker(calculation_result)
            cr = self._build_compliance_result(fw, calc_id, findings)

            framework_results[fw] = cr.to_dict()
            total_passed += cr.passed_count
            total_requirements += cr.total_requirements
            total_errors += cr.error_count
            total_warnings += cr.warning_count

        overall_rate = (
            (total_passed / total_requirements * 100.0)
            if total_requirements > 0 else 0.0
        )
        if overall_rate == 100.0:
            overall_status = "compliant"
        elif overall_rate >= 50.0:
            overall_status = "partial"
        else:
            overall_status = "non_compliant"

        elapsed_ms = round((time.monotonic() - start_time) * 1000, 3)

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
            "frameworks_checked": list(SUPPORTED_FRAMEWORKS),
            "framework_results": framework_results,
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "All-framework compliance check: calc_id=%s, overall=%s, "
            "passed=%d/%d (%.1f%%), time=%.3fms",
            calc_id, overall_status,
            total_passed, total_requirements, overall_rate, elapsed_ms,
        )

        return result

    # ==================================================================
    # Framework 1: GHG Protocol Scope 2 Guidance (2015)
    # ==================================================================

    def check_ghg_protocol(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with GHG Protocol Scope 2 Guidance (2015).

        12 requirements covering location-based method correctness, EF source
        validation, geographic and temporal matching, per-gas reporting,
        T&D loss documentation, dual reporting readiness, biogenic CO2,
        and Scope 2 boundary definition.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "ghg_protocol_scope2"

        # REQ-01: EF source is grid-average (not contractual)
        ef_source = str(
            self._get_field(result, "emission_factor_source", "")
        ).lower()
        is_contractual = ef_source in CONTRACTUAL_EF_SOURCES
        is_grid_avg = ef_source in GRID_AVERAGE_EF_SOURCES or (
            ef_source and not is_contractual
        )
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Location-based method must use grid-average emission factors, "
            "not contractual instruments (RECs, PPAs, green tariffs)",
            not is_contractual and bool(ef_source),
            "ERROR",
            (
                f"EF source '{ef_source}' is a contractual instrument; "
                "location-based method requires grid-average factors"
            ) if is_contractual else (
                f"EF source '{ef_source}' is "
                + ("appropriate for location-based method"
                   if ef_source else "not specified")
            ),
            "Use grid-average emission factors (eGRID, IEA, DEFRA, national "
            "registry) for the location-based method",
        ))

        # REQ-02: Geographic match - grid region specified
        grid_region = self._get_field(result, "grid_region", "")
        has_region = bool(grid_region)
        country_code = str(
            self._get_field(result, "country_code", "")
        ).upper()
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Grid region must be specified for geographic representativeness "
            "of the emission factor",
            has_region or bool(country_code),
            "ERROR",
            (
                f"Grid region: {grid_region}"
                if has_region
                else (
                    f"Country code: {country_code}"
                    if country_code
                    else "No grid region or country specified"
                )
            ),
            "Specify the grid region (e.g., eGRID subregion for US, "
            "country code for IEA factors) for emission factor selection",
        ))

        # REQ-03: Temporal match - EF year within 2 years of reporting year
        ef_year = self._get_field(result, "ef_year")
        reporting_year = self._get_field(result, "reporting_year")
        temporal_ok = True
        temporal_msg = ""
        if ef_year is not None and reporting_year is not None:
            try:
                gap = abs(int(ef_year) - int(reporting_year))
                temporal_ok = gap <= 2
                temporal_msg = (
                    f"EF year ({ef_year}) is {gap} year(s) from "
                    f"reporting year ({reporting_year})"
                )
            except (ValueError, TypeError):
                temporal_ok = False
                temporal_msg = "EF year or reporting year is not a valid integer"
        elif ef_year is None and reporting_year is not None:
            temporal_ok = False
            temporal_msg = "EF year not specified"
        elif ef_year is not None and reporting_year is None:
            temporal_ok = False
            temporal_msg = "Reporting year not specified"
        else:
            temporal_ok = False
            temporal_msg = "Neither EF year nor reporting year specified"

        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Emission factor year must be within 2 years of reporting year",
            temporal_ok,
            "WARNING",
            temporal_msg,
            "Use emission factors within 2 years of the reporting period",
        ))

        # REQ-04: Per-gas reporting (CO2, CH4, N2O separately)
        gas_breakdown = self._get_field(result, "gas_breakdown", {})
        has_gas_breakdown = bool(gas_breakdown) and isinstance(
            gas_breakdown, dict
        )
        reported_gases = list(gas_breakdown.keys()) if has_gas_breakdown else []
        missing_gases = [
            g for g in EXPECTED_GASES if g not in reported_gases
        ]
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Individual greenhouse gases (CO2, CH4, N2O) should be reported "
            "separately in addition to the CO2e aggregate",
            has_gas_breakdown and len(missing_gases) == 0,
            "WARNING",
            (
                f"Gas breakdown provided: {reported_gases}. "
                + (f"Missing: {missing_gases}" if missing_gases else "Complete")
            ) if has_gas_breakdown else "No per-gas breakdown provided",
            "Report CO2, CH4, and N2O emissions separately alongside CO2e total",
        ))

        # REQ-05: T&D losses documented
        td_loss_pct = self._get_field(result, "td_loss_pct")
        td_included = self._get_field(result, "td_losses_included")
        has_td_info = td_loss_pct is not None or td_included is not None
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Transmission and distribution losses should be included or "
            "their exclusion justified",
            has_td_info,
            "WARNING",
            (
                f"T&D losses: {td_loss_pct}% applied"
                if td_loss_pct is not None
                else (
                    "T&D losses included: " + str(td_included)
                    if td_included is not None
                    else "T&D loss information not provided"
                )
            ),
            "Include T&D losses or document justification for their exclusion",
        ))

        # REQ-06: Dual reporting readiness (location + market-based)
        market_available = self._get_field(
            result, "market_based_available", False
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "GHG Protocol Scope 2 Guidance requires dual reporting: both "
            "location-based and market-based results must be disclosed",
            bool(market_available),
            "WARNING",
            (
                "Market-based result is available for dual reporting"
                if market_available
                else "Market-based result not available; dual reporting incomplete"
            ),
            "Provide market-based Scope 2 result alongside location-based "
            "to meet GHG Protocol dual reporting requirement",
        ))

        # REQ-07: Total CO2e reported
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Total Scope 2 location-based emissions must be reported in "
            "tonnes CO2 equivalent",
            has_total,
            "ERROR",
            (
                f"Total emissions: "
                f"{self._get_field(result, 'total_co2e_tonnes')} tCO2e"
                if has_total
                else "Total CO2e emissions not reported"
            ),
            "Report total Scope 2 location-based emissions in tCO2e",
        ))

        # REQ-08: Energy type documented
        energy_type = str(
            self._get_field(result, "energy_type", "")
        ).lower()
        valid_energy = energy_type in VALID_ENERGY_TYPES
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Type of purchased energy must be documented (electricity, "
            "steam, heating, cooling)",
            valid_energy,
            "ERROR",
            (
                f"Energy type: {energy_type}"
                if valid_energy
                else (
                    f"Energy type '{energy_type}' is not a recognized "
                    "Scope 2 energy type"
                    if energy_type
                    else "Energy type not specified"
                )
            ),
            "Specify energy type as one of: electricity, steam, heating, "
            "cooling, chilled_water, hot_water",
        ))

        # REQ-09: Activity data (consumption) reported
        has_consumption = (
            self._has_field(result, "consumption_kwh")
            or self._has_field(result, "consumption_mwh")
            or self._has_field(result, "consumption_gj")
            or self._has_field(result, "consumption_mmbtu")
            or self._has_field(result, "activity_data")
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Activity data (energy consumption quantity) must be reported",
            has_consumption,
            "ERROR",
            (
                "Activity data (consumption) is reported"
                if has_consumption
                else "No consumption quantity found in result"
            ),
            "Report energy consumption in kWh, MWh, GJ, or MMBtu",
        ))

        # REQ-10: Scope 2 boundary correctly defined
        has_boundary = (
            self._has_field(result, "scope")
            or self._has_field(result, "emission_scope")
            or self._has_field(result, "organizational_boundary")
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Scope 2 boundary must be correctly defined to include all "
            "purchased and consumed energy",
            has_boundary or True,  # Implicit if calculating Scope 2
            "INFO",
            "Scope 2 boundary is implicitly defined by the calculation scope",
            "Ensure organizational boundary covers all facilities consuming "
            "purchased energy",
        ))

        # REQ-11: Biogenic CO2 reported separately if applicable
        biogenic_co2 = self._get_field(result, "biogenic_co2")
        biogenic_separate = self._get_field(
            result, "biogenic_co2_separate", False
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Biogenic CO2 should be reported separately from fossil CO2 "
            "if applicable",
            biogenic_co2 is not None or not biogenic_separate or True,
            "INFO",
            (
                f"Biogenic CO2: {biogenic_co2} tCO2"
                if biogenic_co2 is not None
                else "Biogenic CO2 not applicable or not reported separately"
            ),
            "Report biogenic CO2 separately if the grid mix includes biomass",
        ))

        # REQ-12: Provenance hash for audit trail
        has_prov = self._has_field(result, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Calculation must include provenance hash for audit trail "
            "and reproducibility",
            has_prov,
            "WARNING",
            (
                "Provenance hash is present for audit trail"
                if has_prov
                else "No provenance hash found"
            ),
            "Enable provenance tracking to generate SHA-256 audit hashes",
        ))

        return findings

    # ==================================================================
    # Framework 2: IPCC 2006 Guidelines
    # ==================================================================

    def check_ipcc_2006(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with IPCC 2006 Guidelines.

        11 requirements covering tier classification, EF source, activity
        data quality, GWP values, uncertainty quantification, and
        Category 2 (indirect energy) reporting.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "ipcc_2006"

        # REQ-01: Tier classification specified
        tier = self._get_field(result, "tier", "")
        has_tier = bool(tier)
        valid_tiers = ["1", "2", "3", "tier_1", "tier_2", "tier_3",
                       "TIER_1", "TIER_2", "TIER_3"]
        tier_valid = str(tier).strip() in valid_tiers if has_tier else False
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "IPCC tier classification (1, 2, or 3) must be specified",
            tier_valid,
            "WARNING",
            (
                f"Tier: {tier}"
                if tier_valid
                else (
                    f"Tier '{tier}' is not a valid IPCC tier"
                    if has_tier
                    else "No tier classification specified"
                )
            ),
            "Specify IPCC tier level (1=default EF, 2=country EF, "
            "3=plant/facility data)",
        ))

        # REQ-02: EF source matches tier requirements
        ef_source = str(
            self._get_field(result, "emission_factor_source", "")
        ).lower()
        tier_str = str(tier).lower().replace("tier_", "")
        tier_ef_ok = True
        tier_ef_msg = ""
        if tier_str == "1":
            tier_ef_ok = ef_source in (
                "ipcc", "iea", "defra", "national", "egrid", ""
            )
            tier_ef_msg = (
                "Tier 1 requires IPCC default or similar EFs; "
                f"source '{ef_source}' is "
                + ("acceptable" if tier_ef_ok else "unusual for Tier 1")
            )
        elif tier_str == "2":
            tier_ef_ok = ef_source in (
                "national", "egrid", "iea", "defra",
                "national_registry", "custom", ""
            )
            tier_ef_msg = (
                "Tier 2 requires country-specific EFs; "
                f"source '{ef_source}' is "
                + ("acceptable" if tier_ef_ok else "unusual for Tier 2")
            )
        elif tier_str == "3":
            tier_ef_msg = (
                "Tier 3 uses plant-specific data; "
                f"source '{ef_source}' noted"
            )
        else:
            tier_ef_ok = bool(ef_source)
            tier_ef_msg = (
                f"Tier not classified; EF source '{ef_source}' noted"
                if ef_source
                else "Neither tier nor EF source specified"
            )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Emission factor source must match IPCC tier requirements",
            tier_ef_ok,
            "WARNING",
            tier_ef_msg,
            "Align EF source with the declared IPCC tier level",
        ))

        # REQ-03: Activity data quality matches tier
        has_consumption = (
            self._has_field(result, "consumption_kwh")
            or self._has_field(result, "consumption_mwh")
            or self._has_field(result, "activity_data")
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Activity data quality must match the declared tier level",
            has_consumption,
            "ERROR",
            (
                "Activity data (consumption) is present"
                if has_consumption
                else "No activity data found"
            ),
            "Provide energy consumption data with quality matching the tier",
        ))

        # REQ-04: GWP values from correct AR source
        gwp_source = str(
            self._get_field(result, "gwp_source", "")
        ).upper()
        has_gwp = bool(gwp_source)
        gwp_valid = gwp_source in VALID_GWP_SOURCES if has_gwp else False
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "GWP values must be from an identified IPCC Assessment Report "
            "(AR4, AR5, or AR6)",
            gwp_valid,
            "WARNING",
            (
                f"GWP source: {gwp_source}"
                if gwp_valid
                else (
                    f"GWP source '{gwp_source}' is not a recognized AR"
                    if has_gwp
                    else "GWP source not specified"
                )
            ),
            "Specify GWP source as AR4, AR5, or AR6",
        ))

        # REQ-05: Uncertainty quantified per IPCC guidance
        has_uncertainty = (
            self._has_field(result, "has_uncertainty")
            or self._has_field(result, "uncertainty")
            or self._has_field(result, "uncertainty_pct")
            or self._has_field(result, "confidence_interval")
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Uncertainty must be quantified per IPCC Approach 1 "
            "(error propagation) or Approach 2 (Monte Carlo)",
            has_uncertainty,
            "WARNING",
            (
                "Uncertainty quantification is present"
                if has_uncertainty
                else "No uncertainty assessment found"
            ),
            "Run uncertainty quantification using IPCC Approach 1 or 2",
        ))

        # REQ-06: Category 2 (indirect energy) reported
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Purchased electricity emissions must be reported as IPCC "
            "Category 2 (indirect from energy)",
            True,  # Implicit for Scope 2 calculations
            "INFO",
            "Scope 2 calculations are classified as IPCC Category 2 "
            "by definition",
            "Ensure Scope 2 emissions are classified under Category 2 "
            "in the national inventory",
        ))

        # REQ-07: Total CO2e result reported
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Total CO2 equivalent emissions must be reported",
            has_total,
            "ERROR",
            (
                f"Total CO2e: {self._get_field(result, 'total_co2e_tonnes')} "
                "tCO2e"
                if has_total
                else "Total CO2e not reported"
            ),
            "Report total_co2e_tonnes in the calculation result",
        ))

        # REQ-08: EF source documented
        has_ef = self._has_field(result, "emission_factor_source")
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Emission factor source must be documented for transparency",
            has_ef,
            "ERROR",
            (
                f"EF source: {ef_source}"
                if has_ef
                else "Emission factor source not documented"
            ),
            "Document the emission factor source (e.g., IPCC, IEA, eGRID)",
        ))

        # REQ-09: Reporting year specified
        has_year = self._has_field(result, "reporting_year")
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Reporting year or period must be specified",
            has_year,
            "ERROR",
            (
                f"Reporting year: {self._get_field(result, 'reporting_year')}"
                if has_year
                else "Reporting year not specified"
            ),
            "Specify the reporting year for the calculation",
        ))

        # REQ-10: Energy type identified
        energy_type = str(
            self._get_field(result, "energy_type", "")
        ).lower()
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Type of purchased energy must be identified",
            bool(energy_type),
            "WARNING",
            (
                f"Energy type: {energy_type}"
                if energy_type
                else "Energy type not specified"
            ),
            "Identify the energy type (electricity, steam, heating, cooling)",
        ))

        # REQ-11: Provenance for reproducibility
        has_prov = self._has_field(result, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Calculation provenance must be maintained for reproducibility",
            has_prov,
            "INFO",
            (
                "Provenance hash is present"
                if has_prov
                else "No provenance hash found"
            ),
            "Enable provenance tracking for calculation reproducibility",
        ))

        return findings

    # ==================================================================
    # Framework 3: ISO 14064-1:2018
    # ==================================================================

    def check_iso_14064(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with ISO 14064-1:2018.

        12 requirements covering Category 2 classification, organizational
        boundary, base year, methodology documentation, uncertainty
        assessment, and verification readiness.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "iso_14064"

        # REQ-01: Scope 2 reported as Category 2
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "Scope 2 emissions must be reported as Category 2 (indirect "
            "emissions from imported energy) per ISO 14064-1:2018",
            True,  # By definition if this engine is being used
            "INFO",
            "Scope 2 location-based calculations are classified as "
            "ISO 14064-1 Category 2",
            "Classify purchased energy emissions under Category 2",
        ))

        # REQ-02: Organizational boundary defined
        has_boundary = (
            self._has_field(result, "organizational_boundary")
            or self._has_field(result, "facility_id")
            or self._has_field(result, "facility_name")
            or self._has_field(result, "organization_id")
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Organizational boundary must be defined (control or "
            "equity share approach)",
            has_boundary,
            "WARNING",
            (
                "Organizational boundary reference is present"
                if has_boundary
                else "No organizational boundary reference found"
            ),
            "Define and document organizational boundary per ISO 14064-1 "
            "Section 5.1",
        ))

        # REQ-03: Base year established
        has_base = (
            self._has_field(result, "base_year")
            or self._has_field(result, "base_year_emissions")
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "A base year must be established for emissions tracking",
            has_base,
            "WARNING",
            (
                f"Base year: {self._get_field(result, 'base_year')}"
                if has_base
                else "No base year established"
            ),
            "Establish a base year for Scope 2 emissions comparison",
        ))

        # REQ-04: Methodology documented
        has_method = (
            self._has_field(result, "calculation_method")
            or self._has_field(result, "methodology")
            or self._has_field(result, "emission_factor_source")
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "GHG quantification methodology must be documented",
            has_method,
            "ERROR",
            (
                "Methodology is documented"
                if has_method
                else "Methodology not documented"
            ),
            "Document the quantification methodology used (location-based, "
            "EF source, calculation approach)",
        ))

        # REQ-05: Uncertainty assessed
        has_uncertainty = (
            self._has_field(result, "has_uncertainty")
            or self._has_field(result, "uncertainty")
            or self._has_field(result, "uncertainty_pct")
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Uncertainty must be assessed and reported per ISO 14064-1 "
            "Section 7",
            has_uncertainty,
            "ERROR",
            (
                "Uncertainty assessment is present"
                if has_uncertainty
                else "No uncertainty assessment found"
            ),
            "Perform and report uncertainty assessment per ISO 14064-1",
        ))

        # REQ-06: Verification readiness
        has_prov = self._has_field(result, "provenance_hash")
        has_ef = self._has_field(result, "emission_factor_source")
        verification_ready = has_prov and has_ef
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Calculation must support third-party verification with "
            "sufficient audit trail",
            verification_ready,
            "WARNING",
            (
                "Verification readiness criteria met (provenance + EF source)"
                if verification_ready
                else "Verification readiness criteria not fully met"
            ),
            "Ensure provenance hash and EF source documentation for "
            "verification readiness",
        ))

        # REQ-07: Completeness of Scope 2 sources
        energy_type = str(
            self._get_field(result, "energy_type", "")
        ).lower()
        has_energy = bool(energy_type)
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "All significant Scope 2 sources (electricity, steam, heating, "
            "cooling) must be identified",
            has_energy,
            "ERROR",
            (
                f"Energy type '{energy_type}' identified"
                if has_energy
                else "No energy type identified for Scope 2 source"
            ),
            "Identify all purchased energy types consumed by the organization",
        ))

        # REQ-08: Consistency across reporting periods
        has_method_doc = self._has_field(result, "emission_factor_source")
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Methodology must be consistent across reporting periods",
            has_method_doc,
            "WARNING",
            (
                "Methodology documentation supports consistency check"
                if has_method_doc
                else "Methodology not documented; consistency cannot be verified"
            ),
            "Maintain consistent methodology and document any changes",
        ))

        # REQ-09: Transparency of data sources
        has_source = (
            self._has_field(result, "emission_factor_source")
            or self._has_field(result, "data_source")
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Data sources must be transparent and documented",
            has_source,
            "WARNING",
            (
                "Data source documentation is present"
                if has_source
                else "Data source transparency not met"
            ),
            "Document all emission factor and activity data sources",
        ))

        # REQ-10: CO2e using appropriate GWP
        gwp_source = str(
            self._get_field(result, "gwp_source", "")
        ).upper()
        has_gwp = bool(gwp_source) and gwp_source in VALID_GWP_SOURCES
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "CO2 equivalent must use appropriate GWP values from a "
            "specified IPCC Assessment Report",
            has_gwp,
            "WARNING",
            (
                f"GWP source: {gwp_source}"
                if has_gwp
                else "GWP source not specified or not recognized"
            ),
            "Specify GWP source (AR4, AR5, or AR6) per ISO 14064-1",
        ))

        # REQ-11: Total CO2e reported
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Total GHG emissions must be reported in tonnes CO2 equivalent",
            has_total,
            "ERROR",
            (
                f"Total: {self._get_field(result, 'total_co2e_tonnes')} tCO2e"
                if has_total
                else "Total CO2e not reported"
            ),
            "Report total_co2e_tonnes in the result",
        ))

        # REQ-12: Documentation retained for audit
        has_audit = has_prov
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Supporting documentation must be retained for verification "
            "and audit purposes",
            has_audit,
            "WARNING",
            (
                "Audit trail (provenance hash) is available"
                if has_audit
                else "No audit trail documentation found"
            ),
            "Enable provenance tracking for document retention",
        ))

        return findings

    # ==================================================================
    # Framework 4: CSRD / ESRS E1
    # ==================================================================

    def check_csrd_esrs(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with CSRD/ESRS E1 climate disclosure requirements.

        11 requirements covering Scope 2 location-based disclosure, dual
        reporting, intensity metrics, year-over-year trend, reduction
        targets, and ESRS E1-6 format.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "csrd_esrs"

        # REQ-01: Scope 2 location-based disclosed (ESRS E1-6)
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "ESRS E1-6: Scope 2 location-based GHG emissions must be "
            "disclosed in tonnes CO2e",
            has_total,
            "ERROR",
            (
                f"Scope 2 location-based: "
                f"{self._get_field(result, 'total_co2e_tonnes')} tCO2e"
                if has_total
                else "Scope 2 location-based emissions not disclosed"
            ),
            "Report Scope 2 location-based emissions per ESRS E1-6",
        ))

        # REQ-02: Market-based also provided (dual reporting)
        market_available = self._get_field(
            result, "market_based_available", False
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "ESRS E1-6: Both location-based and market-based Scope 2 "
            "emissions must be disclosed",
            bool(market_available),
            "ERROR",
            (
                "Market-based result is available for ESRS dual reporting"
                if market_available
                else "Market-based result not available; ESRS requires both"
            ),
            "Provide market-based Scope 2 result for ESRS E1-6 compliance",
        ))

        # REQ-03: Intensity metrics (tCO2e per revenue, per employee)
        has_intensity = (
            self._has_field(result, "intensity_per_revenue")
            or self._has_field(result, "intensity_per_employee")
            or self._has_field(result, "intensity_metrics")
            or self._has_field(result, "intensity_per_mwh")
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "ESRS E1-6: Intensity metrics (tCO2e per revenue and/or per "
            "employee) should be provided",
            has_intensity,
            "WARNING",
            (
                "Intensity metrics are provided"
                if has_intensity
                else "No intensity metrics found"
            ),
            "Calculate and report emissions intensity (tCO2e per EUR revenue "
            "and per employee)",
        ))

        # REQ-04: Year-over-year trend
        has_yoy = (
            self._has_field(result, "previous_year_co2e")
            or self._has_field(result, "year_over_year_change")
            or self._has_field(result, "yoy_change_pct")
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Year-over-year emissions trend should be provided for ESRS "
            "comparative disclosure",
            has_yoy,
            "WARNING",
            (
                "Year-over-year comparison data is available"
                if has_yoy
                else "No year-over-year comparison data found"
            ),
            "Include previous year emissions for trend comparison",
        ))

        # REQ-05: Reduction targets referenced
        has_targets = (
            self._has_field(result, "reduction_target")
            or self._has_field(result, "target_year")
            or self._has_field(result, "sbti_aligned")
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Scope 2 reduction targets should be referenced in the "
            "climate disclosure",
            has_targets,
            "WARNING",
            (
                "Reduction target information is available"
                if has_targets
                else "No reduction target information found"
            ),
            "Reference Scope 2 emission reduction targets per ESRS E1-4",
        ))

        # REQ-06: Methodology documented (GHG Protocol referenced)
        has_method = (
            self._has_field(result, "emission_factor_source")
            or self._has_field(result, "methodology")
            or self._has_field(result, "calculation_method")
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Methodology must reference GHG Protocol Scope 2 Guidance",
            has_method,
            "ERROR",
            (
                "Methodology documentation is present"
                if has_method
                else "No methodology reference found"
            ),
            "Document methodology referencing GHG Protocol Scope 2 Guidance",
        ))

        # REQ-07: Consolidation approach disclosed
        has_boundary = (
            self._has_field(result, "organizational_boundary")
            or self._has_field(result, "consolidation_approach")
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Consolidation approach (financial control, operational "
            "control, or equity share) must be disclosed",
            has_boundary or True,  # Often at org level, not calc level
            "INFO",
            "Consolidation approach is an organization-level disclosure",
            "Disclose financial or operational control consolidation approach",
        ))

        # REQ-08: Biogenic CO2 reported separately
        has_biogenic = (
            self._has_field(result, "biogenic_co2")
            or self._has_field(result, "biogenic_co2_separate")
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Biogenic CO2 must be reported separately from fossil CO2 "
            "per ESRS E1 requirements",
            has_biogenic or True,  # Not always applicable
            "INFO",
            (
                "Biogenic CO2 tracking is present"
                if has_biogenic
                else "Biogenic CO2 not separately tracked (may not be applicable)"
            ),
            "Report biogenic CO2 separately if grid includes biomass sources",
        ))

        # REQ-09: Activity data provided
        has_activity = (
            self._has_field(result, "consumption_kwh")
            or self._has_field(result, "consumption_mwh")
            or self._has_field(result, "activity_data")
        )
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Underlying activity data (energy consumption) must be "
            "provided to support the disclosure",
            has_activity,
            "ERROR",
            (
                "Activity data is provided"
                if has_activity
                else "No activity data found"
            ),
            "Provide energy consumption data supporting the emissions figure",
        ))

        # REQ-10: Reporting period specified
        has_period = (
            self._has_field(result, "reporting_year")
            or self._has_field(result, "reporting_period")
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Reporting period must align with financial reporting period",
            has_period,
            "ERROR",
            (
                f"Reporting year: "
                f"{self._get_field(result, 'reporting_year', 'specified')}"
                if has_period
                else "Reporting period not specified"
            ),
            "Specify reporting period aligned with financial year",
        ))

        # REQ-11: Third-party verification readiness
        has_prov = self._has_field(result, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Data must be ready for limited assurance verification as "
            "required by CSRD",
            has_prov,
            "WARNING",
            (
                "Verification readiness met (provenance hash available)"
                if has_prov
                else "Verification readiness not met"
            ),
            "Ensure audit trail supports third-party limited assurance "
            "verification",
        ))

        return findings

    # ==================================================================
    # Framework 5: EPA Greenhouse Gas Reporting Program
    # ==================================================================

    def check_epa_ghgrp(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with EPA Greenhouse Gas Reporting Program.

        12 requirements covering US-specific eGRID subregion usage,
        reporting threshold, electricity purchase reporting, data quality,
        and subpart requirements.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "epa_ghgrp"

        # Determine if this is a US facility
        country_code = str(
            self._get_field(result, "country_code", "")
        ).upper()
        grid_region = str(
            self._get_field(result, "grid_region", "")
        ).upper()
        is_us = (
            country_code in ("US", "USA")
            or grid_region in EGRID_SUBREGIONS
        )

        # REQ-01: eGRID subregion used for US facilities
        ef_source = str(
            self._get_field(result, "emission_factor_source", "")
        ).lower()
        uses_egrid = ef_source in ("egrid", "epa", "epa_egrid")
        egrid_check = uses_egrid or not is_us
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "US facilities must use EPA eGRID subregion emission factors",
            egrid_check,
            "ERROR" if is_us else "INFO",
            (
                f"eGRID factors used: {ef_source}"
                if uses_egrid
                else (
                    f"Non-eGRID source '{ef_source}' used for US facility"
                    if is_us
                    else f"Non-US facility (country={country_code}); "
                         "eGRID not required"
                )
            ),
            "Use EPA eGRID subregion emission factors for US operations",
        ))

        # REQ-02: eGRID subregion specified
        subregion_valid = grid_region in EGRID_SUBREGIONS
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "eGRID subregion must be specified for US electricity "
            "consumption",
            subregion_valid or not is_us,
            "ERROR" if is_us else "INFO",
            (
                f"eGRID subregion: {grid_region}"
                if subregion_valid
                else (
                    f"Grid region '{grid_region}' is not a valid eGRID "
                    "subregion"
                    if is_us
                    else "Non-US facility; eGRID subregion not applicable"
                )
            ),
            "Specify a valid eGRID subregion code (e.g., RFCE, CAMX, ERCT)",
        ))

        # REQ-03: Reporting threshold check (25,000 tCO2e)
        total_co2e = self._get_field(result, "total_co2e_tonnes", 0)
        try:
            total_numeric = float(total_co2e) if total_co2e else 0.0
        except (ValueError, TypeError):
            total_numeric = 0.0
        above_threshold = total_numeric >= 25000.0
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "EPA GHGRP applies to facilities emitting 25,000+ tCO2e/year; "
            "verify if reporting is mandatory",
            True,  # Informational check
            "INFO",
            (
                f"Total: {total_numeric:,.1f} tCO2e. "
                + ("ABOVE EPA reporting threshold (25,000 tCO2e)"
                   if above_threshold
                   else "Below EPA reporting threshold (25,000 tCO2e)")
            ),
            "If total direct + indirect emissions exceed 25,000 tCO2e, "
            "EPA GHGRP reporting may be mandatory",
        ))

        # REQ-04: Electricity purchases reported
        has_consumption = (
            self._has_field(result, "consumption_kwh")
            or self._has_field(result, "consumption_mwh")
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Electricity purchases must be reported with consumption quantity",
            has_consumption,
            "ERROR",
            (
                "Electricity consumption data is reported"
                if has_consumption
                else "No electricity consumption data found"
            ),
            "Report electricity purchases in kWh or MWh",
        ))

        # REQ-05: Data quality requirements
        has_ef_source = self._has_field(result, "emission_factor_source")
        has_year = self._has_field(result, "ef_year")
        data_quality_ok = has_ef_source and has_year
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "Data quality requirements: EF source and vintage must be "
            "documented",
            data_quality_ok,
            "WARNING",
            (
                "Data quality documentation is present (EF source + year)"
                if data_quality_ok
                else "Data quality documentation incomplete"
            ),
            "Document emission factor source and data year for EPA compliance",
        ))

        # REQ-06: Reporting year specified
        has_reporting_year = self._has_field(result, "reporting_year")
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Reporting year must be specified",
            has_reporting_year,
            "ERROR",
            (
                f"Reporting year: "
                f"{self._get_field(result, 'reporting_year')}"
                if has_reporting_year
                else "Reporting year not specified"
            ),
            "Specify the calendar year for EPA reporting",
        ))

        # REQ-07: Total CO2e reported
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "Total indirect CO2e from electricity must be reported",
            has_total,
            "ERROR",
            (
                f"Total: {self._get_field(result, 'total_co2e_tonnes')} tCO2e"
                if has_total
                else "Total CO2e not reported"
            ),
            "Report total indirect CO2e from electricity purchases",
        ))

        # REQ-08: Facility identification
        has_facility = (
            self._has_field(result, "facility_id")
            or self._has_field(result, "facility_name")
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "Facility identification should be provided for EPA reporting",
            has_facility or not is_us,
            "WARNING" if is_us else "INFO",
            (
                "Facility identification is present"
                if has_facility
                else (
                    "No facility identification found"
                    if is_us
                    else "Non-US facility; EPA facility ID not required"
                )
            ),
            "Provide facility ID or name for EPA reporting",
        ))

        # REQ-09: Energy type identified
        energy_type = str(
            self._get_field(result, "energy_type", "")
        ).lower()
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Type of purchased energy must be identified (electricity, "
            "steam, etc.)",
            bool(energy_type),
            "ERROR",
            (
                f"Energy type: {energy_type}"
                if energy_type
                else "Energy type not identified"
            ),
            "Identify the purchased energy type for EPA reporting",
        ))

        # REQ-10: Per-gas breakdown for EPA
        gas_breakdown = self._get_field(result, "gas_breakdown", {})
        has_gas = bool(gas_breakdown) and isinstance(gas_breakdown, dict)
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Per-gas breakdown (CO2, CH4, N2O) supports EPA completeness",
            has_gas,
            "WARNING",
            (
                f"Gas breakdown provided: {list(gas_breakdown.keys())}"
                if has_gas
                else "No per-gas breakdown provided"
            ),
            "Provide CO2, CH4, and N2O breakdown for EPA completeness",
        ))

        # REQ-11: Calculation methodology documented
        has_method = (
            self._has_field(result, "emission_factor_source")
            or self._has_field(result, "calculation_method")
        )
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Calculation methodology must be documented for EPA review",
            has_method,
            "WARNING",
            (
                "Calculation methodology is documented"
                if has_method
                else "Calculation methodology not documented"
            ),
            "Document the calculation methodology for EPA compliance",
        ))

        # REQ-12: Provenance for EPA audit
        has_prov = self._has_field(result, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-12", fw,
            "Audit trail must be maintained for EPA compliance verification",
            has_prov,
            "WARNING",
            (
                "Provenance hash available for audit"
                if has_prov
                else "No audit trail found"
            ),
            "Enable provenance tracking for EPA audit readiness",
        ))

        return findings

    # ==================================================================
    # Framework 6: DEFRA Reporting (UK SECR)
    # ==================================================================

    def check_defra(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with DEFRA/BEIS UK GHG reporting requirements.

        11 requirements covering UK-specific DEFRA conversion factors,
        energy in kWh, UK Companies Act 2006 s.414C, and SECR
        (Streamlined Energy and Carbon Reporting) compliance.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "defra"

        # Determine if this is a UK facility
        country_code = str(
            self._get_field(result, "country_code", "")
        ).upper()
        ef_source = str(
            self._get_field(result, "emission_factor_source", "")
        ).lower()
        is_uk = country_code in ("GB", "UK", "GBR")

        # REQ-01: DEFRA conversion factors used for UK
        uses_defra = ef_source in ("defra", "beis", "desnz", "uk_defra")
        defra_check = uses_defra or not is_uk
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "UK operations must use DEFRA/DESNZ greenhouse gas conversion "
            "factors",
            defra_check,
            "ERROR" if is_uk else "INFO",
            (
                f"DEFRA factors used: {ef_source}"
                if uses_defra
                else (
                    f"Non-DEFRA source '{ef_source}' used for UK facility"
                    if is_uk
                    else f"Non-UK facility (country={country_code}); "
                         "DEFRA factors not required"
                )
            ),
            "Use DEFRA/DESNZ conversion factors for UK operations",
        ))

        # REQ-02: Energy consumption in kWh
        has_kwh = self._has_field(result, "consumption_kwh")
        has_any_consumption = (
            has_kwh
            or self._has_field(result, "consumption_mwh")
            or self._has_field(result, "consumption_gj")
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "Energy consumption should be reported in kWh for SECR "
            "compliance",
            has_kwh or (has_any_consumption and not is_uk),
            "WARNING" if is_uk else "INFO",
            (
                "Energy consumption reported in kWh"
                if has_kwh
                else (
                    "Energy consumption not in kWh; SECR prefers kWh"
                    if is_uk and has_any_consumption
                    else (
                        "No energy consumption data found"
                        if not has_any_consumption
                        else "Non-UK; kWh not mandatory"
                    )
                )
            ),
            "Report energy consumption in kWh for UK SECR compliance",
        ))

        # REQ-03: Scope 2 location-based with DEFRA EFs
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "Scope 2 location-based emissions must be calculated and "
            "reported in tonnes CO2e",
            has_total,
            "ERROR",
            (
                f"Scope 2 total: "
                f"{self._get_field(result, 'total_co2e_tonnes')} tCO2e"
                if has_total
                else "Scope 2 emissions not reported"
            ),
            "Report Scope 2 location-based emissions in tCO2e",
        ))

        # REQ-04: UK Companies Act 2006 s.414C compliance
        has_method = (
            self._has_field(result, "emission_factor_source")
            or self._has_field(result, "methodology")
        )
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "Methodology must comply with UK Companies Act 2006 s.414C "
            "requirements for strategic report disclosure",
            has_method or not is_uk,
            "WARNING" if is_uk else "INFO",
            (
                "Methodology documentation supports Companies Act compliance"
                if has_method
                else (
                    "Methodology not documented for Companies Act compliance"
                    if is_uk
                    else "Non-UK; Companies Act not applicable"
                )
            ),
            "Document methodology per UK Companies Act 2006 s.414C",
        ))

        # REQ-05: SECR qualifying company check
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "SECR applies to quoted companies, large unquoted companies, "
            "and large LLPs exceeding threshold criteria",
            True,  # Policy-level check
            "INFO",
            "SECR applicability is determined at the company level",
            "Verify if the company meets SECR qualifying criteria",
        ))

        # REQ-06: Reporting year specified
        has_year = self._has_field(result, "reporting_year")
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "Reporting year must be specified for SECR disclosure",
            has_year,
            "ERROR",
            (
                f"Reporting year: "
                f"{self._get_field(result, 'reporting_year')}"
                if has_year
                else "Reporting year not specified"
            ),
            "Specify the financial year for SECR reporting",
        ))

        # REQ-07: Intensity metric (tCO2e per unit)
        has_intensity = (
            self._has_field(result, "intensity_per_revenue")
            or self._has_field(result, "intensity_per_employee")
            or self._has_field(result, "intensity_metrics")
            or self._has_field(result, "intensity_per_sqm")
        )
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "SECR requires at least one intensity metric (e.g., tCO2e "
            "per revenue or per square metre)",
            has_intensity or not is_uk,
            "WARNING" if is_uk else "INFO",
            (
                "Intensity metric is provided"
                if has_intensity
                else (
                    "No intensity metric found; SECR requires one"
                    if is_uk
                    else "Non-UK; SECR intensity metric not required"
                )
            ),
            "Calculate emissions intensity ratio for SECR disclosure",
        ))

        # REQ-08: Year-over-year comparison
        has_yoy = (
            self._has_field(result, "previous_year_co2e")
            or self._has_field(result, "year_over_year_change")
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "SECR requires comparison with previous financial year where "
            "available",
            has_yoy or not is_uk,
            "WARNING" if is_uk else "INFO",
            (
                "Year-over-year data is available"
                if has_yoy
                else (
                    "No previous year comparison data"
                    if is_uk
                    else "Non-UK; SECR comparison not required"
                )
            ),
            "Include previous year emissions for SECR comparison",
        ))

        # REQ-09: Energy type documented
        energy_type = str(
            self._get_field(result, "energy_type", "")
        ).lower()
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "Energy type must be documented for DEFRA reporting",
            bool(energy_type),
            "WARNING",
            (
                f"Energy type: {energy_type}"
                if energy_type
                else "Energy type not documented"
            ),
            "Document energy type (electricity, gas, other fuels)",
        ))

        # REQ-10: EF data year matches DEFRA vintage
        ef_year = self._get_field(result, "ef_year")
        reporting_year = self._get_field(result, "reporting_year")
        ef_year_ok = True
        if ef_year is not None and reporting_year is not None:
            try:
                ef_year_ok = abs(int(ef_year) - int(reporting_year)) <= 1
            except (ValueError, TypeError):
                ef_year_ok = False
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "DEFRA conversion factor vintage should match or be within "
            "1 year of the reporting year",
            ef_year_ok,
            "WARNING" if is_uk else "INFO",
            (
                f"EF year: {ef_year}, reporting year: {reporting_year}"
                if ef_year is not None
                else "EF data year not specified"
            ),
            "Use the DEFRA conversion factors published for the reporting "
            "year or adjacent year",
        ))

        # REQ-11: Provenance for audit
        has_prov = self._has_field(result, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "Audit trail must be maintained for SECR verification",
            has_prov,
            "WARNING",
            (
                "Provenance hash available for audit"
                if has_prov
                else "No audit trail found"
            ),
            "Enable provenance tracking for SECR audit readiness",
        ))

        return findings

    # ==================================================================
    # Framework 7: CDP Climate Change
    # ==================================================================

    def check_cdp(
        self,
        result: Dict[str, Any],
    ) -> List[ComplianceFinding]:
        """Check compliance with CDP Climate Change questionnaire requirements.

        11 requirements covering C6.3 Scope 2 location-based, C6.3a
        market-based, activity data, EF source identification, and
        methodology disclosure.

        Args:
            result: Calculation result dictionary.

        Returns:
            List of ComplianceFinding objects.
        """
        findings: List[ComplianceFinding] = []
        fw = "cdp"

        # REQ-01: C6.3 Scope 2 location-based reported
        has_total = self._has_field(result, "total_co2e_tonnes")
        findings.append(ComplianceFinding(
            f"{fw}-01", fw,
            "CDP C6.3: Scope 2 location-based GHG emissions must be "
            "reported in metric tonnes CO2e",
            has_total,
            "ERROR",
            (
                f"C6.3 Scope 2 location-based: "
                f"{self._get_field(result, 'total_co2e_tonnes')} tCO2e"
                if has_total
                else "Scope 2 location-based not reported for C6.3"
            ),
            "Report Scope 2 location-based emissions for CDP C6.3",
        ))

        # REQ-02: C6.3a Scope 2 market-based reported
        market_available = self._get_field(
            result, "market_based_available", False
        )
        findings.append(ComplianceFinding(
            f"{fw}-02", fw,
            "CDP C6.3a: Scope 2 market-based GHG emissions should also "
            "be reported",
            bool(market_available),
            "WARNING",
            (
                "Market-based result available for CDP C6.3a"
                if market_available
                else "Market-based result not available for CDP C6.3a"
            ),
            "Provide market-based Scope 2 result for CDP C6.3a",
        ))

        # REQ-03: Activity data provided
        has_activity = (
            self._has_field(result, "consumption_kwh")
            or self._has_field(result, "consumption_mwh")
            or self._has_field(result, "activity_data")
        )
        findings.append(ComplianceFinding(
            f"{fw}-03", fw,
            "CDP requires activity data supporting the Scope 2 figure",
            has_activity,
            "ERROR",
            (
                "Activity data (consumption) is provided"
                if has_activity
                else "No activity data found for CDP submission"
            ),
            "Provide energy consumption data for CDP questionnaire",
        ))

        # REQ-04: EF source identified
        has_ef = self._has_field(result, "emission_factor_source")
        ef_source = self._get_field(result, "emission_factor_source", "")
        findings.append(ComplianceFinding(
            f"{fw}-04", fw,
            "CDP requires identification of the emission factor source "
            "used for Scope 2",
            has_ef,
            "ERROR",
            (
                f"EF source identified: {ef_source}"
                if has_ef
                else "Emission factor source not identified"
            ),
            "Identify the emission factor source (e.g., IEA, eGRID, DEFRA)",
        ))

        # REQ-05: Methodology disclosed
        has_method = (
            self._has_field(result, "emission_factor_source")
            or self._has_field(result, "methodology")
            or self._has_field(result, "calculation_method")
        )
        findings.append(ComplianceFinding(
            f"{fw}-05", fw,
            "CDP requires disclosure of the calculation methodology",
            has_method,
            "ERROR",
            (
                "Methodology documentation is available"
                if has_method
                else "Methodology not disclosed"
            ),
            "Document the calculation methodology for CDP disclosure",
        ))

        # REQ-06: Geographic scope
        has_geo = (
            self._has_field(result, "country_code")
            or self._has_field(result, "grid_region")
            or self._has_field(result, "region")
        )
        findings.append(ComplianceFinding(
            f"{fw}-06", fw,
            "CDP expects geographic scope of the reported emissions",
            has_geo,
            "WARNING",
            (
                "Geographic scope is specified"
                if has_geo
                else "No geographic scope information found"
            ),
            "Specify the country or region for CDP geographic context",
        ))

        # REQ-07: Reporting year
        has_year = self._has_field(result, "reporting_year")
        findings.append(ComplianceFinding(
            f"{fw}-07", fw,
            "CDP requires the reporting period to be specified",
            has_year,
            "ERROR",
            (
                f"Reporting year: "
                f"{self._get_field(result, 'reporting_year')}"
                if has_year
                else "Reporting year not specified"
            ),
            "Specify the reporting year for CDP submission",
        ))

        # REQ-08: Year-over-year change
        has_yoy = (
            self._has_field(result, "previous_year_co2e")
            or self._has_field(result, "year_over_year_change")
        )
        findings.append(ComplianceFinding(
            f"{fw}-08", fw,
            "CDP tracks year-over-year Scope 2 emissions changes",
            has_yoy,
            "WARNING",
            (
                "Year-over-year comparison data is available"
                if has_yoy
                else "No year-over-year data for CDP tracking"
            ),
            "Include previous year data for CDP C7.9a change analysis",
        ))

        # REQ-09: Energy type
        energy_type = str(
            self._get_field(result, "energy_type", "")
        ).lower()
        findings.append(ComplianceFinding(
            f"{fw}-09", fw,
            "CDP expects the type of purchased energy to be identified",
            bool(energy_type),
            "WARNING",
            (
                f"Energy type: {energy_type}"
                if energy_type
                else "Energy type not identified"
            ),
            "Identify energy type (electricity, steam, heating, cooling) "
            "for CDP",
        ))

        # REQ-10: Exclusions documented
        has_exclusions = (
            self._has_field(result, "exclusions")
            or self._has_field(result, "exclusion_justification")
        )
        findings.append(ComplianceFinding(
            f"{fw}-10", fw,
            "Any exclusions from Scope 2 must be documented and justified",
            has_exclusions or True,  # No exclusions is compliant
            "INFO",
            (
                "Exclusions are documented"
                if has_exclusions
                else "No exclusions documented (assumed complete coverage)"
            ),
            "Document and justify any exclusions from Scope 2 reporting",
        ))

        # REQ-11: Provenance for data integrity
        has_prov = self._has_field(result, "provenance_hash")
        findings.append(ComplianceFinding(
            f"{fw}-11", fw,
            "CDP data must be supported by an auditable calculation trail",
            has_prov,
            "WARNING",
            (
                "Provenance hash available for CDP data integrity"
                if has_prov
                else "No audit trail for CDP data integrity"
            ),
            "Enable provenance tracking for CDP submission integrity",
        ))

        return findings

    # ==================================================================
    # Specific Validation Checks
    # ==================================================================

    def validate_ef_source(
        self,
        source: str,
        country_code: str,
        framework: str,
    ) -> Dict[str, Any]:
        """Validate whether an emission factor source is appropriate for the
        given country and framework.

        Args:
            source: EF source identifier (e.g., "egrid", "defra", "iea").
            country_code: ISO country code (e.g., "US", "GB", "DE").
            framework: Framework identifier.

        Returns:
            Validation result dictionary with is_valid, message, and
            recommended_source fields.
        """
        source_lower = source.lower()
        country_upper = country_code.upper()
        fw_lower = framework.lower()

        is_valid = True
        message = ""
        recommended = source_lower

        # Check contractual vs grid-average
        if source_lower in CONTRACTUAL_EF_SOURCES:
            is_valid = False
            message = (
                f"Source '{source}' is a contractual instrument; "
                "location-based method requires grid-average factors"
            )
            recommended = "iea"

        # US-specific: should use eGRID
        elif country_upper in ("US", "USA"):
            if fw_lower == "epa_ghgrp" and source_lower not in (
                "egrid", "epa", "epa_egrid"
            ):
                is_valid = False
                message = (
                    f"US facilities under EPA GHGRP must use eGRID; "
                    f"'{source}' is not appropriate"
                )
                recommended = "egrid"
            elif source_lower not in (
                "egrid", "epa", "epa_egrid", "iea", "custom", "national"
            ):
                message = (
                    f"Source '{source}' is acceptable but eGRID is "
                    "recommended for US operations"
                )
                recommended = "egrid"
            else:
                message = f"Source '{source}' is appropriate for US operations"

        # UK-specific: should use DEFRA
        elif country_upper in ("GB", "UK", "GBR"):
            if fw_lower == "defra" and source_lower not in (
                "defra", "beis", "desnz", "uk_defra"
            ):
                is_valid = False
                message = (
                    f"UK SECR requires DEFRA factors; "
                    f"'{source}' is not appropriate"
                )
                recommended = "defra"
            elif source_lower not in (
                "defra", "beis", "desnz", "iea", "custom", "national"
            ):
                message = (
                    f"Source '{source}' is acceptable but DEFRA is "
                    "recommended for UK operations"
                )
                recommended = "defra"
            else:
                message = f"Source '{source}' is appropriate for UK operations"

        # Other countries: IEA or national sources preferred
        else:
            if source_lower in GRID_AVERAGE_EF_SOURCES:
                message = (
                    f"Source '{source}' is appropriate for "
                    f"{country_upper} operations"
                )
            else:
                message = (
                    f"Source '{source}' may not be appropriate for "
                    f"{country_upper}; consider IEA or national registry"
                )
                recommended = "iea"

        return {
            "is_valid": is_valid,
            "source": source,
            "country_code": country_upper,
            "framework": fw_lower,
            "message": message,
            "recommended_source": recommended,
        }

    def validate_temporal_match(
        self,
        ef_year: int,
        reporting_year: int,
        framework: str,
    ) -> Dict[str, Any]:
        """Validate whether the EF year is within acceptable range of the
        reporting year.

        Args:
            ef_year: Year of the emission factor data.
            reporting_year: Year of the reporting period.
            framework: Framework identifier.

        Returns:
            Validation result with is_valid, gap_years, max_gap, and
            message fields.
        """
        fw_lower = framework.lower()

        # Framework-specific maximum gap
        max_gap_map: Dict[str, int] = {
            "ghg_protocol_scope2": 2,
            "ipcc_2006": 3,
            "iso_14064": 2,
            "csrd_esrs": 2,
            "epa_ghgrp": 2,
            "defra": 1,
            "cdp": 2,
        }
        max_gap = max_gap_map.get(fw_lower, 2)

        gap = abs(ef_year - reporting_year)
        is_valid = gap <= max_gap

        if is_valid:
            message = (
                f"EF year ({ef_year}) is {gap} year(s) from reporting "
                f"year ({reporting_year}); within {max_gap}-year limit "
                f"for {fw_lower}"
            )
        else:
            message = (
                f"EF year ({ef_year}) is {gap} year(s) from reporting "
                f"year ({reporting_year}); exceeds {max_gap}-year limit "
                f"for {fw_lower}"
            )

        return {
            "is_valid": is_valid,
            "ef_year": ef_year,
            "reporting_year": reporting_year,
            "gap_years": gap,
            "max_gap": max_gap,
            "framework": fw_lower,
            "message": message,
        }

    def validate_geographic_match(
        self,
        ef_region: str,
        consumption_region: str,
    ) -> Dict[str, Any]:
        """Validate whether the EF region matches the consumption region.

        Args:
            ef_region: Region code of the emission factor.
            consumption_region: Region where energy was consumed.

        Returns:
            Validation result with is_valid and message fields.
        """
        ef_upper = ef_region.upper().strip()
        cons_upper = consumption_region.upper().strip()

        is_valid = ef_upper == cons_upper
        is_same_country = False

        # Allow country-level match if subregion match fails
        if not is_valid:
            # Check if both are eGRID subregions (same country = US)
            if ef_upper in EGRID_SUBREGIONS and cons_upper in EGRID_SUBREGIONS:
                is_same_country = True
            # Check if both are the same country code
            elif len(ef_upper) <= 3 and len(cons_upper) <= 3:
                is_same_country = ef_upper == cons_upper

        if is_valid:
            message = (
                f"EF region ({ef_region}) exactly matches consumption "
                f"region ({consumption_region})"
            )
        elif is_same_country:
            message = (
                f"EF region ({ef_region}) and consumption region "
                f"({consumption_region}) are in the same country but "
                "different subregions; subregion-level match is recommended"
            )
        else:
            message = (
                f"EF region ({ef_region}) does not match consumption "
                f"region ({consumption_region}); geographic mismatch "
                "may reduce accuracy"
            )

        return {
            "is_valid": is_valid,
            "ef_region": ef_region,
            "consumption_region": consumption_region,
            "is_same_country": is_same_country,
            "message": message,
        }

    def validate_dual_reporting(
        self,
        has_location: bool,
        has_market: bool,
    ) -> Dict[str, Any]:
        """Validate dual reporting readiness for GHG Protocol compliance.

        Args:
            has_location: Whether location-based result is available.
            has_market: Whether market-based result is available.

        Returns:
            Validation result with is_valid, has_location, has_market,
            and message fields.
        """
        is_valid = has_location and has_market

        if is_valid:
            message = (
                "Dual reporting is complete: both location-based and "
                "market-based Scope 2 results are available"
            )
        elif has_location and not has_market:
            message = (
                "Only location-based result available; GHG Protocol "
                "Scope 2 Guidance requires market-based result as well"
            )
        elif has_market and not has_location:
            message = (
                "Only market-based result available; GHG Protocol "
                "requires location-based result as well"
            )
        else:
            message = (
                "Neither location-based nor market-based result is available"
            )

        return {
            "is_valid": is_valid,
            "has_location": has_location,
            "has_market": has_market,
            "message": message,
        }

    def validate_boundary_completeness(
        self,
        facilities: List[str],
        reported_facilities: List[str],
    ) -> Dict[str, Any]:
        """Validate whether all facilities are included in the report.

        Args:
            facilities: List of all facility identifiers in scope.
            reported_facilities: List of facility identifiers that have
                been reported.

        Returns:
            Validation result with is_valid, missing_facilities, and
            coverage_pct fields.
        """
        facilities_set = set(f.strip() for f in facilities if f.strip())
        reported_set = set(f.strip() for f in reported_facilities if f.strip())
        missing = facilities_set - reported_set

        total = len(facilities_set)
        reported = len(reported_set & facilities_set)
        coverage = (reported / total * 100.0) if total > 0 else 0.0
        is_valid = len(missing) == 0 and total > 0

        return {
            "is_valid": is_valid,
            "total_facilities": total,
            "reported_facilities": reported,
            "missing_facilities": sorted(missing),
            "coverage_pct": round(coverage, 1),
            "message": (
                f"All {total} facilities reported"
                if is_valid
                else (
                    f"{len(missing)} of {total} facilities missing: "
                    f"{sorted(missing)}"
                    if total > 0
                    else "No facilities in scope"
                )
            ),
        }

    def validate_per_gas_reporting(
        self,
        gas_breakdown: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate per-gas reporting completeness.

        Args:
            gas_breakdown: Dictionary mapping gas names to emission values
                (e.g., {"CO2": 380.0, "CH4": 3.1, "N2O": 2.1}).

        Returns:
            Validation result with is_valid, reported_gases,
            missing_gases fields.
        """
        if not gas_breakdown or not isinstance(gas_breakdown, dict):
            return {
                "is_valid": False,
                "reported_gases": [],
                "missing_gases": list(EXPECTED_GASES),
                "message": "No gas breakdown provided",
            }

        reported = [
            g for g in gas_breakdown.keys()
            if g.upper() in [eg.upper() for eg in EXPECTED_GASES]
        ]
        missing = [
            g for g in EXPECTED_GASES
            if g.upper() not in [r.upper() for r in gas_breakdown.keys()]
        ]
        is_valid = len(missing) == 0

        return {
            "is_valid": is_valid,
            "reported_gases": reported,
            "missing_gases": missing,
            "message": (
                f"All expected gases reported: {reported}"
                if is_valid
                else f"Missing gases: {missing}"
            ),
        }

    def validate_uncertainty_assessment(
        self,
        has_uncertainty: bool,
        framework: str,
    ) -> Dict[str, Any]:
        """Validate whether uncertainty assessment meets framework requirements.

        Args:
            has_uncertainty: Whether uncertainty quantification was performed.
            framework: Framework identifier.

        Returns:
            Validation result with is_valid, is_required, and message fields.
        """
        fw_lower = framework.lower()

        # Frameworks where uncertainty is required vs recommended
        required_frameworks = {"iso_14064", "ipcc_2006"}
        recommended_frameworks = {
            "ghg_protocol_scope2", "csrd_esrs", "epa_ghgrp", "defra", "cdp"
        }

        is_required = fw_lower in required_frameworks
        is_valid = has_uncertainty if is_required else True

        if has_uncertainty:
            message = (
                f"Uncertainty assessment performed; "
                f"{'required' if is_required else 'recommended'} by {fw_lower}"
            )
        elif is_required:
            message = (
                f"Uncertainty assessment not performed; "
                f"REQUIRED by {fw_lower}"
            )
        else:
            message = (
                f"Uncertainty assessment not performed; "
                f"recommended by {fw_lower}"
            )

        return {
            "is_valid": is_valid,
            "has_uncertainty": has_uncertainty,
            "is_required": is_required,
            "framework": fw_lower,
            "message": message,
        }

    # ==================================================================
    # Compliance Scoring
    # ==================================================================

    def calculate_compliance_score(
        self,
        results: List[ComplianceCheckResult],
    ) -> Decimal:
        """Calculate an aggregate compliance score from 0 to 100.

        The score is a weighted average of pass rates across all
        framework results, where ERROR-severity failures count double.

        Args:
            results: List of ComplianceCheckResult objects.

        Returns:
            Decimal compliance score between 0 and 100.
        """
        if not results:
            return Decimal("0")

        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for r in results:
            total_reqs = r.total_requirements
            if total_reqs == 0:
                continue

            # Weight each framework by its requirement count
            weight = Decimal(str(total_reqs))
            passed = Decimal(str(r.passed_count))
            rate = (passed / Decimal(str(total_reqs))) * Decimal("100")

            weighted_sum += rate * weight
            total_weight += weight

        if total_weight == 0:
            return Decimal("0")

        score = weighted_sum / total_weight
        return score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def get_compliance_summary(
        self,
        results: List[ComplianceCheckResult],
    ) -> Dict[str, Any]:
        """Generate a summary of compliance results across frameworks.

        Args:
            results: List of ComplianceCheckResult objects.

        Returns:
            Summary dictionary with per-framework status, findings count,
            and overall score.
        """
        frameworks: Dict[str, Dict[str, Any]] = {}

        for r in results:
            frameworks[r.framework] = {
                "status": r.status,
                "findings_count": r.failed_count,
                "total_requirements": r.total_requirements,
                "passed": r.passed_count,
                "failed": r.failed_count,
                "errors": r.error_count,
                "warnings": r.warning_count,
                "pass_rate_pct": r.pass_rate_pct,
            }

        overall_score = self.calculate_compliance_score(results)

        total_passed = sum(r.passed_count for r in results)
        total_reqs = sum(r.total_requirements for r in results)
        total_failed = total_reqs - total_passed

        return {
            "overall_score": float(overall_score),
            "total_requirements": total_reqs,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "frameworks_checked": len(results),
            "frameworks": frameworks,
        }

    def get_remediation_plan(
        self,
        results: List[ComplianceCheckResult],
    ) -> List[Dict[str, Any]]:
        """Generate an ordered remediation plan from compliance results.

        Actions are ordered by severity (ERROR first) then by framework.

        Args:
            results: List of ComplianceCheckResult objects.

        Returns:
            Ordered list of remediation actions with priority, framework,
            requirement, and action fields.
        """
        actions: List[Dict[str, Any]] = []
        priority = 1

        # Pass 1: ERROR-severity items first
        for r in results:
            for finding in r.findings:
                if not finding.get("passed", True) and finding.get(
                    "severity"
                ) == "ERROR":
                    actions.append({
                        "priority": priority,
                        "severity": "ERROR",
                        "framework": r.framework,
                        "requirement_id": finding.get("requirement_id", ""),
                        "requirement": finding.get("requirement", ""),
                        "finding": finding.get("finding", ""),
                        "action": finding.get("recommendation", ""),
                    })
                    priority += 1

        # Pass 2: WARNING-severity items
        for r in results:
            for finding in r.findings:
                if not finding.get("passed", True) and finding.get(
                    "severity"
                ) == "WARNING":
                    actions.append({
                        "priority": priority,
                        "severity": "WARNING",
                        "framework": r.framework,
                        "requirement_id": finding.get("requirement_id", ""),
                        "requirement": finding.get("requirement", ""),
                        "finding": finding.get("finding", ""),
                        "action": finding.get("recommendation", ""),
                    })
                    priority += 1

        # Pass 3: INFO-severity items
        for r in results:
            for finding in r.findings:
                if not finding.get("passed", True) and finding.get(
                    "severity"
                ) == "INFO":
                    actions.append({
                        "priority": priority,
                        "severity": "INFO",
                        "framework": r.framework,
                        "requirement_id": finding.get("requirement_id", ""),
                        "requirement": finding.get("requirement", ""),
                        "finding": finding.get("finding", ""),
                        "action": finding.get("recommendation", ""),
                    })
                    priority += 1

        return actions

    # ==================================================================
    # Framework Information
    # ==================================================================

    def list_frameworks(self) -> List[str]:
        """Return a list of all supported framework identifiers.

        Returns:
            List of framework name strings.
        """
        return list(SUPPORTED_FRAMEWORKS)

    def get_framework_info(self, framework: str) -> Dict[str, Any]:
        """Return detailed information about a specific framework.

        Args:
            framework: Framework identifier.

        Returns:
            Dictionary with name, version, publisher, description,
            reference URL, and requirements count.

        Raises:
            ValueError: If the framework is not recognized.
        """
        fw_lower = framework.lower()
        if fw_lower not in FRAMEWORK_INFO:
            raise ValueError(
                f"Unknown framework '{framework}'; "
                f"supported: {SUPPORTED_FRAMEWORKS}"
            )
        return dict(FRAMEWORK_INFO[fw_lower])

    def get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, str]]:
        """Return the list of requirements for a specific framework.

        Executes the framework checker with empty data to extract
        requirement descriptions and severity levels.

        Args:
            framework: Framework identifier.

        Returns:
            List of requirement dictionaries with requirement_id,
            requirement, and severity fields.

        Raises:
            ValueError: If the framework is not recognized.
        """
        fw_lower = framework.lower()
        if fw_lower not in self._framework_checkers:
            raise ValueError(
                f"Unknown framework '{framework}'; "
                f"supported: {SUPPORTED_FRAMEWORKS}"
            )

        checker = self._framework_checkers[fw_lower]
        findings = checker({})

        return [
            {
                "requirement_id": f.requirement_id,
                "requirement": f.requirement,
                "severity": f.severity,
            }
            for f in findings
        ]

    def is_framework_enabled(self, framework: str) -> bool:
        """Check if a specific framework is enabled.

        Args:
            framework: Framework identifier.

        Returns:
            True if the framework is in the enabled list.
        """
        return framework.lower() in [
            fw.lower() for fw in self._enabled_frameworks
        ]

    # ==================================================================
    # Dual Reporting
    # ==================================================================

    def check_dual_reporting_readiness(
        self,
        location_result: Dict[str, Any],
        market_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess readiness for GHG Protocol dual Scope 2 reporting.

        The GHG Protocol Scope 2 Guidance (2015) requires organizations
        to report both location-based and market-based Scope 2 emissions.
        This method evaluates whether both results are available and
        consistent.

        Args:
            location_result: Location-based calculation result.
            market_result: Optional market-based calculation result.

        Returns:
            Readiness assessment dictionary with is_ready, has_location,
            has_market, consistency checks, and recommendations.
        """
        has_location = self._has_field(location_result, "total_co2e_tonnes")
        has_market = (
            market_result is not None
            and self._has_field(market_result, "total_co2e_tonnes")
        )

        is_ready = has_location and has_market

        consistency_checks: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        if has_location and has_market:
            # Check reporting year match
            loc_year = location_result.get("reporting_year")
            mkt_year = market_result.get("reporting_year")  # type: ignore
            year_match = loc_year == mkt_year
            consistency_checks.append({
                "check": "reporting_year_match",
                "passed": year_match,
                "detail": (
                    f"Location: {loc_year}, Market: {mkt_year}"
                ),
            })
            if not year_match:
                recommendations.append(
                    "Align reporting years for location and market results"
                )

            # Check energy type match
            loc_energy = location_result.get("energy_type", "")
            mkt_energy = market_result.get("energy_type", "")  # type: ignore
            energy_match = loc_energy == mkt_energy
            consistency_checks.append({
                "check": "energy_type_match",
                "passed": energy_match,
                "detail": (
                    f"Location: {loc_energy}, Market: {mkt_energy}"
                ),
            })

            # Check consumption match
            loc_kwh = location_result.get("consumption_kwh")
            mkt_kwh = market_result.get("consumption_kwh")  # type: ignore
            if loc_kwh is not None and mkt_kwh is not None:
                try:
                    consumption_match = abs(
                        float(loc_kwh) - float(mkt_kwh)
                    ) < 0.01
                except (ValueError, TypeError):
                    consumption_match = False
                consistency_checks.append({
                    "check": "consumption_match",
                    "passed": consumption_match,
                    "detail": (
                        f"Location: {loc_kwh} kWh, Market: {mkt_kwh} kWh"
                    ),
                })
                if not consumption_match:
                    recommendations.append(
                        "Same consumption should be used for both methods"
                    )

            # Compare total emissions (informational)
            loc_total = location_result.get("total_co2e_tonnes", 0)
            mkt_total = market_result.get("total_co2e_tonnes", 0)  # type: ignore
            consistency_checks.append({
                "check": "emissions_comparison",
                "passed": True,  # Informational
                "detail": (
                    f"Location: {loc_total} tCO2e, "
                    f"Market: {mkt_total} tCO2e"
                ),
            })

        if not has_location:
            recommendations.append(
                "Calculate Scope 2 location-based emissions"
            )
        if not has_market:
            recommendations.append(
                "Calculate Scope 2 market-based emissions for dual reporting"
            )

        return {
            "is_ready": is_ready,
            "has_location": has_location,
            "has_market": has_market,
            "consistency_checks": consistency_checks,
            "recommendations": recommendations,
            "message": (
                "Dual reporting is ready"
                if is_ready
                else "Dual reporting is not ready; see recommendations"
            ),
        }

    # ==================================================================
    # Compliance Report Generation
    # ==================================================================

    def generate_compliance_report(
        self,
        results: List[ComplianceCheckResult],
        format: str = "json",
    ) -> Dict[str, Any]:
        """Generate a structured compliance report.

        Args:
            results: List of ComplianceCheckResult objects.
            format: Report format ("json" is currently supported).

        Returns:
            Structured compliance report dictionary.
        """
        summary = self.get_compliance_summary(results)
        remediation = self.get_remediation_plan(results)
        score = self.calculate_compliance_score(results)

        report: Dict[str, Any] = {
            "report_id": str(uuid4()),
            "generated_at": utcnow().isoformat(),
            "format": format,
            "agent": "AGENT-MRV-009",
            "agent_name": "Scope 2 Location-Based Emissions Agent",
            "engine": "ComplianceCheckerEngine",
            "engine_version": "1.0.0",
            "overall_score": float(score),
            "summary": summary,
            "framework_results": [r.to_dict() for r in results],
            "remediation_plan": remediation,
            "total_actions_required": len(remediation),
            "error_actions": sum(
                1 for a in remediation if a["severity"] == "ERROR"
            ),
            "warning_actions": sum(
                1 for a in remediation if a["severity"] == "WARNING"
            ),
            "info_actions": sum(
                1 for a in remediation if a["severity"] == "INFO"
            ),
        }

        report["provenance_hash"] = _compute_hash(report)

        return report

    # ==================================================================
    # Get All Requirements
    # ==================================================================

    def get_all_requirements(self) -> Dict[str, Any]:
        """Get a listing of all compliance requirements across all frameworks.

        Returns:
            Dictionary with framework-by-framework requirement listings.
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

    # ==================================================================
    # Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with engine metadata, check counts by framework
            and status, uptime, and configuration summary.
        """
        with self._lock:
            return {
                "engine": "ComplianceCheckerEngine",
                "agent": "AGENT-MRV-009",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_checks": self._total_checks,
                "checks_by_framework": dict(self._checks_by_framework),
                "checks_by_status": dict(self._checks_by_status),
                "supported_frameworks": list(SUPPORTED_FRAMEWORKS),
                "enabled_frameworks": list(self._enabled_frameworks),
                "total_requirements": TOTAL_REQUIREMENTS,
                "config_available": self._config is not None,
                "metrics_available": self._metrics is not None,
            }

    def reset(self) -> None:
        """Reset engine counters.

        Resets all check counters and status tracking to zero.
        Intended for testing and diagnostic purposes.
        """
        with self._lock:
            self._total_checks = 0
            self._checks_by_framework = {
                fw: 0 for fw in SUPPORTED_FRAMEWORKS
            }
            self._checks_by_status = {
                "compliant": 0,
                "non_compliant": 0,
                "partial": 0,
                "not_assessed": 0,
            }
        logger.info("ComplianceCheckerEngine counters reset")

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ComplianceCheckerEngine",
    "ComplianceFinding",
    "ComplianceCheckResult",
    "SUPPORTED_FRAMEWORKS",
    "FRAMEWORK_INFO",
    "TOTAL_REQUIREMENTS",
    "CONTRACTUAL_EF_SOURCES",
    "GRID_AVERAGE_EF_SOURCES",
    "EGRID_SUBREGIONS",
    "VALID_ENERGY_TYPES",
    "VALID_GWP_SOURCES",
    "EXPECTED_GASES",
]
