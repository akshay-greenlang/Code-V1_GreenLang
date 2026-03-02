# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-019 Engine 6

This module implements regulatory compliance checking for Business Travel
emissions (GHG Protocol Scope 3 Category 6) against 7 regulatory frameworks.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 6 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 Climate Change
4. CDP Climate Change Questionnaire (Module 7)
5. SBTi (Science Based Targets initiative)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 Emissions Standard

Category 6-Specific Compliance Rules:
- Radiative forcing (RF) dual reporting (with/without)
- Mode breakdown disclosure
- Double-counting prevention (7 rules: DC-BT-001 through DC-BT-007)
- Category boundary enforcement (Cat 6 vs Cat 7 vs Scope 1)
- Data quality scoring
- Materiality threshold (SBTi: 1% of Scope 3)
- Verification / assurance opinion tracking

Double-Counting Prevention Rules:
    DC-BT-001: No company-owned vehicles (should be Scope 1)
    DC-BT-002: No commuting trips (should be Category 7)
    DC-BT-003: No freight trips (should be Category 4 or 9)
    DC-BT-004: No overlap with upstream transportation (Category 4)
    DC-BT-005: No overlap with Scope 1 fleet
    DC-BT-006: Hotel room-only emissions (exclude food/beverage)
    DC-BT-007: WTT not double-counted with Category 3 (Fuel & Energy Activities)

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-006
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.business_travel.models import (
    ComplianceFramework,
    ComplianceStatus,
    ComplianceCheckInput,
    ComplianceCheckResult,
    RFOption,
    TransportMode,
    FRAMEWORK_REQUIRED_DISCLOSURES,
    calculate_provenance_hash,
)
from greenlang.business_travel.config import get_config

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP


# ==============================================================================
# ENUMS
# ==============================================================================


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories that could overlap with Category 6."""

    SCOPE_1 = "SCOPE_1"  # Company-owned vehicles
    CATEGORY_3 = "CATEGORY_3"  # Fuel & energy activities (WTT)
    CATEGORY_4 = "CATEGORY_4"  # Upstream transportation
    CATEGORY_7 = "CATEGORY_7"  # Employee commuting
    CATEGORY_9 = "CATEGORY_9"  # Downstream transportation


class BoundaryClassification(str, Enum):
    """Category boundary classification for a trip."""

    CATEGORY_6 = "CATEGORY_6"  # Business travel (correct)
    CATEGORY_7 = "CATEGORY_7"  # Employee commuting
    SCOPE_1 = "SCOPE_1"  # Company-owned vehicle
    CATEGORY_4 = "CATEGORY_4"  # Upstream transportation (freight)
    CATEGORY_9 = "CATEGORY_9"  # Downstream transportation (freight)
    EXCLUDED = "EXCLUDED"  # Not in scope


# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass
class ComplianceFinding:
    """Single compliance finding with rule code and severity."""

    rule_code: str
    description: str
    severity: ComplianceSeverity
    framework: str
    status: ComplianceStatus = ComplianceStatus.FAIL
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    regulation_reference: Optional[str] = None


@dataclass
class FrameworkCheckState:
    """Internal state accumulator for a single framework check."""

    framework: ComplianceFramework
    findings: List[ComplianceFinding] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0

    def add_pass(self, rule_code: str, description: str) -> None:
        """Record a passed check."""
        self.passed_checks += 1
        self.total_checks += 1

    def add_fail(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity,
        recommendation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        regulation_reference: Optional[str] = None,
    ) -> None:
        """Record a failed check with a finding."""
        self.failed_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_code=rule_code,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.FAIL,
                details=details,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
            )
        )

    def add_warning(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity = ComplianceSeverity.MEDIUM,
        recommendation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        regulation_reference: Optional[str] = None,
    ) -> None:
        """Record a warning check with a finding."""
        self.warning_checks += 1
        self.total_checks += 1
        self.findings.append(
            ComplianceFinding(
                rule_code=rule_code,
                description=description,
                severity=severity,
                framework=self.framework.value,
                status=ComplianceStatus.WARNING,
                details=details,
                recommendation=recommendation,
                regulation_reference=regulation_reference,
            )
        )

    def compute_score(self) -> Decimal:
        """
        Compute compliance score (0-100) based on pass/fail/warning ratio.

        Weights:
            - Each failed check reduces score proportionally
            - Warnings reduce score at 50% of a failure
            - Perfect score = 100 (all checks passed)
        """
        if self.total_checks == 0:
            return Decimal("100.00")

        # Each fail = 1.0 penalty, each warning = 0.5 penalty
        penalty_points = (
            Decimal(str(self.failed_checks))
            + Decimal(str(self.warning_checks)) * Decimal("0.5")
        )
        max_points = Decimal(str(self.total_checks))
        score = (
            (max_points - penalty_points) / max_points * Decimal("100")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        # Clamp to 0-100
        if score < Decimal("0"):
            score = Decimal("0.00")
        if score > Decimal("100"):
            score = Decimal("100.00")

        return score

    def compute_status(self) -> ComplianceStatus:
        """Compute overall status from findings."""
        if self.failed_checks > 0:
            return ComplianceStatus.FAIL
        if self.warning_checks > 0:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS

    def to_result(self) -> ComplianceCheckResult:
        """Convert accumulated state to a ComplianceCheckResult model."""
        findings_dicts = [
            {
                "rule_code": f.rule_code,
                "description": f.description,
                "severity": f.severity.value,
                "status": f.status.value,
                "recommendation": f.recommendation,
                "regulation_reference": f.regulation_reference,
                "details": f.details,
            }
            for f in self.findings
        ]

        recommendations = [
            f.recommendation
            for f in self.findings
            if f.recommendation is not None
        ]

        return ComplianceCheckResult(
            framework=self.framework,
            status=self.compute_status(),
            score=self.compute_score(),
            findings=findings_dicts,
            recommendations=recommendations,
        )


# ==============================================================================
# SCORING WEIGHTS PER FRAMEWORK
# ==============================================================================

# Weights determine relative importance of each framework in overall score
FRAMEWORK_WEIGHTS: Dict[ComplianceFramework, Decimal] = {
    ComplianceFramework.GHG_PROTOCOL: Decimal("1.00"),
    ComplianceFramework.ISO_14064: Decimal("0.85"),
    ComplianceFramework.CSRD_ESRS: Decimal("0.90"),
    ComplianceFramework.CDP: Decimal("0.85"),
    ComplianceFramework.SBTI: Decimal("0.80"),
    ComplianceFramework.SB_253: Decimal("0.75"),
    ComplianceFramework.GRI: Decimal("0.70"),
}


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for Business Travel emissions (Category 6).

    Validates calculation results against 7 regulatory frameworks with
    Category 6-specific rules including radiative forcing disclosure,
    double-counting prevention, and category boundary enforcement.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _config: Business travel configuration (compliance section)
        _enabled_frameworks: Set of enabled compliance frameworks
        _check_count: Running count of compliance checks performed

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> results = engine.check_all_frameworks(calc_result)
        >>> summary = engine.get_compliance_summary(results)
        >>> print(f"Overall: {summary['overall_status']}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with configuration."""
        self._config = get_config()
        self._enabled_frameworks: List[str] = (
            self._config.compliance.get_frameworks()
        )
        self._check_count: int = 0

        logger.info(
            "ComplianceCheckerEngine initialized: version=%s, "
            "frameworks=%d, strict_mode=%s",
            ENGINE_VERSION,
            len(self._enabled_frameworks),
            self._config.compliance.strict_mode,
        )

    @classmethod
    def get_instance(cls) -> "ComplianceCheckerEngine":
        """
        Get singleton instance (thread-safe double-checked locking).

        Returns:
            ComplianceCheckerEngine singleton instance.

        Example:
            >>> engine = ComplianceCheckerEngine.get_instance()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        Thread Safety:
            Protected by the class-level lock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("ComplianceCheckerEngine singleton reset")

    # ==========================================================================
    # Main Entry Points
    # ==========================================================================

    def check_all_frameworks(
        self, result: dict
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Run all enabled framework checks and return results.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked.

        Args:
            result: Calculation result dictionary containing total_co2e,
                mode_breakdown, rf values, method, ef_sources, etc.

        Returns:
            Dictionary mapping framework name string to ComplianceCheckResult.

        Example:
            >>> all_results = engine.check_all_frameworks(calc_result)
            >>> ghg_result = all_results.get("ghg_protocol")
            >>> ghg_result.status
            <ComplianceStatus.PASS: 'pass'>
        """
        start_time = time.monotonic()
        logger.info("Running compliance checks for all enabled frameworks")

        all_results: Dict[str, ComplianceCheckResult] = {}

        # Map config framework names to check methods
        framework_dispatch: Dict[str, Tuple[ComplianceFramework, str]] = {
            "GHG_PROTOCOL_SCOPE3": (
                ComplianceFramework.GHG_PROTOCOL,
                "check_ghg_protocol",
            ),
            "ISO_14064": (
                ComplianceFramework.ISO_14064,
                "check_iso_14064",
            ),
            "CSRD_ESRS_E1": (
                ComplianceFramework.CSRD_ESRS,
                "check_csrd_esrs",
            ),
            "CDP": (ComplianceFramework.CDP, "check_cdp"),
            "SBTI": (ComplianceFramework.SBTI, "check_sbti"),
            "SB_253": (ComplianceFramework.SB_253, "check_sb_253"),
            "GRI": (ComplianceFramework.GRI, "check_gri"),
            # Also accept model enum values directly
            "ghg_protocol": (
                ComplianceFramework.GHG_PROTOCOL,
                "check_ghg_protocol",
            ),
            "iso_14064": (
                ComplianceFramework.ISO_14064,
                "check_iso_14064",
            ),
            "csrd_esrs": (
                ComplianceFramework.CSRD_ESRS,
                "check_csrd_esrs",
            ),
            "cdp": (ComplianceFramework.CDP, "check_cdp"),
            "sbti": (ComplianceFramework.SBTI, "check_sbti"),
            "sb_253": (ComplianceFramework.SB_253, "check_sb_253"),
            "gri": (ComplianceFramework.GRI, "check_gri"),
            # Also map DEFRA_BEIS to GRI as a placeholder
            "DEFRA_BEIS": (ComplianceFramework.GRI, "check_gri"),
        }

        for framework_name in self._enabled_frameworks:
            dispatch = framework_dispatch.get(framework_name)
            if dispatch is None:
                logger.warning(
                    "Unknown framework in config: '%s', skipping",
                    framework_name,
                )
                continue

            framework_enum, method_name = dispatch
            try:
                check_method = getattr(self, method_name)
                check_result = check_method(result)
                all_results[framework_enum.value] = check_result

                logger.info(
                    "%s compliance: %s (score: %s)",
                    framework_enum.value,
                    check_result.status.value,
                    check_result.score,
                )

            except Exception as e:
                logger.error(
                    "Error checking %s compliance: %s",
                    framework_name,
                    str(e),
                    exc_info=True,
                )
                all_results[framework_enum.value] = ComplianceCheckResult(
                    framework=framework_enum,
                    status=ComplianceStatus.FAIL,
                    score=Decimal("0"),
                    findings=[
                        {
                            "rule_code": "CHECK_ERROR",
                            "description": f"Compliance check failed: {str(e)}",
                            "severity": ComplianceSeverity.CRITICAL.value,
                            "status": ComplianceStatus.FAIL.value,
                        }
                    ],
                    recommendations=[
                        "Resolve the compliance check error and rerun."
                    ],
                )

        duration = time.monotonic() - start_time
        self._check_count += 1

        logger.info(
            "All framework checks complete: %d frameworks, duration=%.4fs",
            len(all_results),
            duration,
        )

        return all_results

    # ==========================================================================
    # Framework: GHG Protocol Scope 3
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 6).

        Checks:
            - total_co2e present and > 0
            - Calculation method documented
            - Emission factor sources documented
            - Exclusions documented
            - DQI score present
            - RF disclosure (if air travel included)

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for GHG Protocol.

        Example:
            >>> res = engine.check_ghg_protocol(calc_result)
            >>> res.framework
            <ComplianceFramework.GHG_PROTOCOL: 'ghg_protocol'>
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # GHG-BT-001: Total emissions present and positive
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GHG-BT-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-BT-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Ensure total_co2e is calculated and > 0.",
                regulation_reference="GHG Protocol Scope 3, Ch 6",
            )

        # GHG-BT-002: Calculation method documented
        method = result.get("method") or result.get("calculation_method")
        if method:
            state.add_pass("GHG-BT-002", "Calculation method documented")
        else:
            state.add_fail(
                "GHG-BT-002",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used "
                    "(supplier-specific, distance-based, average-data, or spend-based)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 6.1",
            )

        # GHG-BT-003: Emission factor sources documented
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GHG-BT-003", "Emission factor sources documented")
        else:
            state.add_fail(
                "GHG-BT-003",
                "Emission factor sources not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document all emission factor sources "
                    "(DEFRA, ICAO, EPA, EEIO, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 6",
            )

        # GHG-BT-004: Exclusions documented
        exclusions = result.get("exclusions")
        if exclusions is not None:
            state.add_pass("GHG-BT-004", "Exclusions documented")
        else:
            state.add_warning(
                "GHG-BT-004",
                "Exclusions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document any exclusions from Category 6 reporting "
                    "(e.g., short-distance trips, certain modes)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 6",
            )

        # GHG-BT-005: Data quality indicator score
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("GHG-BT-005", "DQI score present")
        else:
            state.add_warning(
                "GHG-BT-005",
                "Data quality indicator (DQI) score not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across 5 dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-BT-006: RF disclosure for air travel
        mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
        has_air = False
        if isinstance(mode_breakdown, dict):
            has_air = "air" in mode_breakdown or TransportMode.AIR.value in mode_breakdown

        if has_air:
            with_rf = result.get("total_co2e_with_rf") or result.get("with_rf")
            without_rf = (
                result.get("total_co2e_without_rf") or result.get("without_rf")
            )
            if with_rf is not None and without_rf is not None:
                state.add_pass(
                    "GHG-BT-006",
                    "RF disclosure: both with-RF and without-RF figures present",
                )
            elif with_rf is not None or without_rf is not None:
                state.add_warning(
                    "GHG-BT-006",
                    "Only one RF figure disclosed (need both with-RF and without-RF)",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Report both with-RF and without-RF emissions "
                        "for aviation as recommended by GHG Protocol."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 6",
                )
            else:
                state.add_warning(
                    "GHG-BT-006",
                    "No RF disclosure for aviation emissions",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Disclose radiative forcing impact for aviation. "
                        "Report both with-RF and without-RF figures."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 6",
                )

        return state.to_result()

    # ==========================================================================
    # Framework: ISO 14064
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018.

        Checks:
            - total_co2e present
            - Uncertainty analysis present
            - Base year documented
            - Methodology described

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for ISO 14064.

        Example:
            >>> res = engine.check_iso_14064(calc_result)
            >>> res.framework.value
            'iso_14064'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14064)

        # ISO-BT-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("ISO-BT-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-BT-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total CO2e emissions.",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-BT-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-BT-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-BT-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis "
                    "(Monte Carlo, analytical, or IPCC Tier 2)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-BT-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-BT-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-BT-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison "
                    "and trend analysis."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-BT-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-BT-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-BT-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, data sources, and calculation approach."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-BT-005: Reporting period defined
        reporting_period = (
            result.get("reporting_period")
            or result.get("period")
        )
        if reporting_period:
            state.add_pass("ISO-BT-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-BT-005",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation="Specify the reporting period (e.g., 2024, 2024-Q3).",
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CSRD / ESRS E1
    # ==========================================================================

    def check_csrd_esrs(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CSRD ESRS E1 Climate Change.

        Checks:
            - total_co2e by category
            - Methodology description
            - Targets documented
            - Mode breakdown present

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for CSRD/ESRS.

        Example:
            >>> res = engine.check_csrd_esrs(calc_result)
            >>> res.framework.value
            'csrd_esrs'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CSRD_ESRS)

        # CSRD-BT-001: Total emissions by category
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CSRD-BT-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-BT-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Scope 3 Category 6 emissions.",
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-BT-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-BT-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-BT-002",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for business travel "
                    "emissions as required by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-BT-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-BT-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-BT-003",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for business travel "
                    "(e.g., reduce flights by X%, shift to rail)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-BT-004: Mode breakdown present
        mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
        if mode_breakdown and isinstance(mode_breakdown, dict) and len(mode_breakdown) > 0:
            state.add_pass("CSRD-BT-004", "Mode breakdown present")
        else:
            state.add_fail(
                "CSRD-BT-004",
                "Mode breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by transport mode "
                    "(air, rail, road, hotel, etc.)."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-BT-005: Actions described
        actions = result.get("actions") or result.get("reduction_actions")
        if actions:
            state.add_pass("CSRD-BT-005", "Reduction actions described")
        else:
            state.add_warning(
                "CSRD-BT-005",
                "Reduction actions not described",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Describe actions taken or planned to reduce business "
                    "travel emissions (e.g., virtual meeting policy, "
                    "rail-first policy)."
                ),
                regulation_reference="ESRS E1-3, para 29",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CDP Climate Change
    # ==========================================================================

    def check_cdp(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change Questionnaire.

        CRITICAL CDP Requirements for Business Travel:
            CR-BT-001: Both with-RF and without-RF figures MUST be present
            CR-BT-003: Mode breakdown MUST be present
            Verification status MUST be documented

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for CDP.

        Example:
            >>> res = engine.check_cdp(calc_result)
            >>> res.framework.value
            'cdp'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CDP)

        # CDP-BT-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CDP-BT-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CDP-BT-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 6 emissions to CDP.",
                regulation_reference="CDP CC Module 7, Q7.3a",
            )

        # CR-BT-001: Radiative forcing - BOTH with_rf and without_rf REQUIRED
        with_rf = result.get("total_co2e_with_rf") or result.get("with_rf")
        without_rf = (
            result.get("total_co2e_without_rf") or result.get("without_rf")
        )

        if with_rf is not None and without_rf is not None:
            state.add_pass(
                "CR-BT-001",
                "Both with-RF and without-RF figures present",
            )
        elif with_rf is not None or without_rf is not None:
            # Only one figure present - WARNING
            state.add_warning(
                "CR-BT-001",
                "Only one RF figure present; CDP requires both with-RF and without-RF",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Report BOTH with-RF and without-RF figures for aviation. "
                    "CDP requires dual reporting for Category 6."
                ),
                regulation_reference="CDP CC Module 7, Q7.3a",
            )
        else:
            # Neither present - FAIL if air travel is included
            mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
            has_air = False
            if isinstance(mode_breakdown, dict):
                has_air = (
                    "air" in mode_breakdown
                    or TransportMode.AIR.value in mode_breakdown
                )

            if has_air:
                state.add_fail(
                    "CR-BT-001",
                    "No RF figures disclosed but air travel is included",
                    ComplianceSeverity.CRITICAL,
                    recommendation=(
                        "Report both with-RF and without-RF figures for aviation. "
                        "This is a critical CDP requirement."
                    ),
                    regulation_reference="CDP CC Module 7, Q7.3a",
                )
            else:
                state.add_pass(
                    "CR-BT-001",
                    "No air travel; RF disclosure not applicable",
                )

        # CR-BT-003: Mode breakdown REQUIRED
        mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
        if mode_breakdown and isinstance(mode_breakdown, dict) and len(mode_breakdown) > 0:
            state.add_pass("CR-BT-003", "Mode breakdown present")
        else:
            state.add_fail(
                "CR-BT-003",
                "Mode breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by transport mode "
                    "for CDP reporting."
                ),
                regulation_reference="CDP CC Module 7, Q7.3a",
            )

        # CDP-BT-004: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-BT-004", "Verification status documented")
        else:
            state.add_warning(
                "CDP-BT-004",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether business travel emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module 10, Q10.1",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SBTi
    # ==========================================================================

    def check_sbti(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative.

        Checks:
            - CR-BT-002: RF included in target boundary
            - Target coverage documented
            - Progress tracking present
            - Materiality assessment

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for SBTi.

        Example:
            >>> res = engine.check_sbti(calc_result)
            >>> res.framework.value
            'sbti'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SBTI)

        # SBTI-BT-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SBTI-BT-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-BT-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate total Category 6 emissions for SBTi target boundary.",
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # CR-BT-002: RF included in target boundary
        with_rf = result.get("total_co2e_with_rf") or result.get("with_rf")
        rf_inclusion = result.get("rf_inclusion") or result.get("rf_in_target")
        if rf_inclusion or with_rf is not None:
            state.add_pass(
                "CR-BT-002",
                "Radiative forcing included in target boundary",
            )
        else:
            state.add_warning(
                "CR-BT-002",
                "RF inclusion in SBTi target boundary not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Include radiative forcing (RF) in the SBTi target boundary "
                    "for aviation emissions. SBTi recommends RF-inclusive reporting."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-BT-003: Target coverage
        target_coverage = (
            result.get("target_coverage")
            or result.get("sbti_coverage")
        )
        if target_coverage is not None:
            state.add_pass("SBTI-BT-003", "Target coverage documented")
        else:
            state.add_warning(
                "SBTI-BT-003",
                "SBTi target coverage not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document what percentage of Scope 3 emissions is covered "
                    "by SBTi targets (minimum 67% required)."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-BT-004: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-BT-004", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-BT-004",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 6 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-BT-005: Materiality assessment
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat6_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                if cat6_pct >= self._config.compliance.materiality_threshold * Decimal("100"):
                    state.add_pass(
                        "SBTI-BT-005",
                        f"Category 6 is material ({cat6_pct:.1f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SBTI-BT-005",
                        f"Category 6 below materiality threshold ({cat6_pct:.1f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 6 may not need to be included in SBTi "
                            "target boundary if below materiality threshold."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-BT-005",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-BT-005",
                "Total Scope 3 emissions not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 6 materiality."
                ),
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SB 253
    # ==========================================================================

    def check_sb_253(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Checks:
            - Total CO2e present
            - Methodology documented
            - CR-BT-009: Assurance opinion available
            - CR-BT-004: Materiality > 1% threshold

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for SB 253.

        Example:
            >>> res = engine.check_sb_253(calc_result)
            >>> res.framework.value
            'sb_253'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # SB253-BT-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SB253-BT-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-BT-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 6 emissions for SB 253 compliance.",
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-BT-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-BT-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-BT-002",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # CR-BT-009: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("CR-BT-009", "Assurance opinion available")
        else:
            state.add_warning(
                "CR-BT-009",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # CR-BT-004: Materiality > 1% threshold
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat6_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                if cat6_pct > Decimal("1"):
                    state.add_pass(
                        "CR-BT-004",
                        f"Category 6 exceeds 1% materiality ({cat6_pct:.2f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "CR-BT-004",
                        f"Category 6 below 1% materiality threshold ({cat6_pct:.2f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 6 is below 1% of total Scope 3. "
                            "Consider whether separate reporting is warranted."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "CR-BT-004",
                    "Could not calculate materiality",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "CR-BT-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 6 materiality against 1% threshold."
                ),
            )

        return state.to_result()

    # ==========================================================================
    # Framework: GRI 305
    # ==========================================================================

    def check_gri(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305 Emissions Standard.

        Checks:
            - Total CO2e in metric tonnes
            - Gases included documented
            - Base year present
            - Standards referenced

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for GRI.

        Example:
            >>> res = engine.check_gri(calc_result)
            >>> res.framework.value
            'gri'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GRI)

        # GRI-BT-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GRI-BT-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-BT-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 6 emissions in metric tonnes CO2e.",
                regulation_reference="GRI 305-3",
            )

        # GRI-BT-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-BT-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-BT-002",
                "Gases included not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, or CO2e aggregate)."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-BT-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-BT-003", "Base year present")
        else:
            state.add_warning(
                "GRI-BT-003",
                "Base year not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-BT-004: Standards referenced
        standards = (
            result.get("standards_used")
            or result.get("standards")
            or result.get("framework_references")
        )
        if standards:
            state.add_pass("GRI-BT-004", "Standards referenced")
        else:
            state.add_warning(
                "GRI-BT-004",
                "Standards used not referenced",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Reference the standards and methodologies used "
                    "(e.g., GHG Protocol Scope 3, DEFRA conversion factors)."
                ),
                regulation_reference="GRI 305-3(e)",
            )

        # GRI-BT-005: Source of emission factors
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-BT-005", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-BT-005",
                "Emission factor sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., DEFRA 2024, ICAO 2024)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention
    # ==========================================================================

    def check_double_counting(
        self, trips: list
    ) -> List[dict]:
        """
        Validate 7 double-counting prevention rules for business travel.

        Rules:
            DC-BT-001: No company-owned vehicles (should be Scope 1)
            DC-BT-002: No commuting trips (should be Category 7)
            DC-BT-003: No freight trips (should be Category 4 or 9)
            DC-BT-004: No overlap with upstream transportation (Category 4)
            DC-BT-005: No overlap with Scope 1 fleet
            DC-BT-006: Hotel room-only (no food/beverage emissions)
            DC-BT-007: WTT not double-counted with Category 3

        Args:
            trips: List of trip dictionaries, each containing fields like
                mode, purpose, vehicle_ownership, trip_type, includes_meals,
                wtt_separately_reported, etc.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and affected trip indices.

        Example:
            >>> findings = engine.check_double_counting(trips)
            >>> len(findings)
            0  # No double-counting issues
        """
        findings: List[dict] = []

        for idx, trip in enumerate(trips):
            # DC-BT-001: No company-owned vehicles
            vehicle_ownership = trip.get("vehicle_ownership", "").lower()
            if vehicle_ownership in ("company_owned", "company", "fleet"):
                findings.append({
                    "rule_code": "DC-BT-001",
                    "description": (
                        f"Trip {idx}: Company-owned vehicle detected. "
                        "Should be reported under Scope 1, not Category 6."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "trip_index": idx,
                    "recommendation": (
                        "Exclude company-owned vehicles from Category 6. "
                        "Report under Scope 1 (mobile combustion)."
                    ),
                })

            # DC-BT-002: No commuting trips
            purpose = trip.get("purpose", "").lower()
            trip_type = trip.get("trip_type", "").lower()
            if purpose in ("commuting", "commute") or trip_type == "commute":
                findings.append({
                    "rule_code": "DC-BT-002",
                    "description": (
                        f"Trip {idx}: Commuting trip detected. "
                        "Should be reported under Category 7, not Category 6."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "trip_index": idx,
                    "recommendation": (
                        "Exclude commuting trips from Category 6. "
                        "Report under Category 7 (Employee Commuting)."
                    ),
                })

            # DC-BT-003: No freight trips
            if trip_type in ("freight", "cargo", "logistics", "shipment"):
                findings.append({
                    "rule_code": "DC-BT-003",
                    "description": (
                        f"Trip {idx}: Freight/cargo trip detected. "
                        "Should be Category 4 or 9, not Category 6."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "trip_index": idx,
                    "recommendation": (
                        "Exclude freight from Category 6. "
                        "Report under Category 4 (upstream) or Category 9 (downstream)."
                    ),
                })

            # DC-BT-004: No overlap with Category 4
            also_in_cat4 = trip.get("reported_in_cat4", False)
            if also_in_cat4:
                findings.append({
                    "rule_code": "DC-BT-004",
                    "description": (
                        f"Trip {idx}: Also reported in Category 4 "
                        "(upstream transportation). Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "trip_index": idx,
                    "recommendation": (
                        "Ensure trip is reported in only one category. "
                        "Business travel goes in Category 6; freight goes in Category 4."
                    ),
                })

            # DC-BT-005: No overlap with Scope 1 fleet
            also_in_scope1 = trip.get("reported_in_scope1", False)
            if also_in_scope1:
                findings.append({
                    "rule_code": "DC-BT-005",
                    "description": (
                        f"Trip {idx}: Also reported in Scope 1 fleet. "
                        "Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "trip_index": idx,
                    "recommendation": (
                        "If vehicle is company-owned/leased, report in Scope 1. "
                        "Only third-party vehicles belong in Category 6."
                    ),
                })

            # DC-BT-006: Hotel room-only (no food)
            mode = trip.get("mode", "").lower()
            includes_meals = trip.get("includes_meals", False)
            if mode == "hotel" and includes_meals:
                findings.append({
                    "rule_code": "DC-BT-006",
                    "description": (
                        f"Trip {idx}: Hotel emissions include meals/food. "
                        "Category 6 hotel should be room-only."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "trip_index": idx,
                    "recommendation": (
                        "Separate hotel room emissions from food/beverage. "
                        "Only room-night emissions belong in Category 6 hotel."
                    ),
                })

            # DC-BT-007: WTT not double-counted with Category 3
            wtt_separate = trip.get("wtt_separately_reported", False)
            wtt_in_cat3 = trip.get("wtt_reported_in_cat3", False)
            if wtt_separate and wtt_in_cat3:
                findings.append({
                    "rule_code": "DC-BT-007",
                    "description": (
                        f"Trip {idx}: WTT emissions reported in both Category 6 "
                        "and Category 3. Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "trip_index": idx,
                    "recommendation": (
                        "Report WTT in either Category 3 (Fuel & Energy Activities) "
                        "or as a memo item in Category 6, not both."
                    ),
                })

        logger.info(
            "Double-counting check: %d trips, %d findings",
            len(trips),
            len(findings),
        )

        return findings

    # ==========================================================================
    # Category Boundary
    # ==========================================================================

    def check_category_boundary(
        self, trip: dict
    ) -> Dict[str, Any]:
        """
        Determine if a trip belongs in Category 6, Category 7, Scope 1, or other.

        Classification Rules:
            - Company-owned vehicle -> Scope 1
            - Employee commuting -> Category 7
            - Freight/cargo -> Category 4 or 9
            - Business travel (third-party vehicle) -> Category 6
            - All other -> Excluded

        Args:
            trip: Trip dictionary with mode, purpose, vehicle_ownership,
                trip_type fields.

        Returns:
            Dictionary with classification, reason, and any warnings.

        Example:
            >>> result = engine.check_category_boundary({
            ...     "mode": "air",
            ...     "purpose": "client_visit",
            ...     "vehicle_ownership": "third_party"
            ... })
            >>> result["classification"]
            'CATEGORY_6'
        """
        vehicle_ownership = trip.get("vehicle_ownership", "").lower()
        purpose = trip.get("purpose", "").lower()
        trip_type = trip.get("trip_type", "").lower()
        mode = trip.get("mode", "").lower()

        warnings: List[str] = []

        # Rule 1: Company-owned vehicles -> Scope 1
        if vehicle_ownership in ("company_owned", "company", "fleet", "leased"):
            return {
                "classification": BoundaryClassification.SCOPE_1.value,
                "reason": (
                    "Company-owned/leased vehicle emissions belong in Scope 1 "
                    "(mobile combustion), not Category 6."
                ),
                "warnings": warnings,
                "trip": trip,
            }

        # Rule 2: Commuting -> Category 7
        if purpose in ("commuting", "commute") or trip_type == "commute":
            return {
                "classification": BoundaryClassification.CATEGORY_7.value,
                "reason": (
                    "Employee commuting belongs in Category 7, "
                    "not Category 6 (Business Travel)."
                ),
                "warnings": warnings,
                "trip": trip,
            }

        # Rule 3: Freight -> Category 4 or 9
        if trip_type in ("freight", "cargo", "logistics", "shipment"):
            classification = BoundaryClassification.CATEGORY_4.value
            if trip.get("direction", "").lower() == "downstream":
                classification = BoundaryClassification.CATEGORY_9.value
            return {
                "classification": classification,
                "reason": (
                    "Freight/cargo transportation does not belong in "
                    "Category 6 (Business Travel)."
                ),
                "warnings": warnings,
                "trip": trip,
            }

        # Rule 4: Business travel with third-party vehicle -> Category 6
        valid_business_purposes = {
            "business", "conference", "client_visit",
            "training", "meeting", "site_visit", "other",
        }
        if purpose in valid_business_purposes or not purpose:
            # Check for potential ambiguities
            if mode == "hotel" and trip.get("includes_meals"):
                warnings.append(
                    "Hotel emissions should be room-only. "
                    "Meal emissions may need separate treatment."
                )

            return {
                "classification": BoundaryClassification.CATEGORY_6.value,
                "reason": (
                    "Business travel using third-party transportation "
                    "belongs in Category 6."
                ),
                "warnings": warnings,
                "trip": trip,
            }

        # Default: Excluded
        return {
            "classification": BoundaryClassification.EXCLUDED.value,
            "reason": (
                f"Trip with purpose '{purpose}' and type '{trip_type}' "
                "could not be classified into a Scope 3 category."
            ),
            "warnings": warnings,
            "trip": trip,
        }

    # ==========================================================================
    # Summary & Recommendations
    # ==========================================================================

    def get_compliance_summary(
        self, results: Dict[str, ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Summarize all framework results with overall score and recommendations.

        Calculates a weighted average compliance score across all checked
        frameworks and aggregates findings by severity.

        Args:
            results: Dictionary mapping framework name to ComplianceCheckResult.

        Returns:
            Summary dictionary with overall_score, overall_status,
            framework_scores, findings by severity, and recommendations.

        Example:
            >>> summary = engine.get_compliance_summary(all_results)
            >>> summary["overall_score"]
            85.5
            >>> summary["overall_status"]
            'warning'
        """
        if not results:
            return {
                "overall_score": 0.0,
                "overall_status": ComplianceStatus.FAIL.value,
                "frameworks_checked": 0,
                "framework_scores": {},
                "critical_findings": [],
                "high_findings": [],
                "medium_findings": [],
                "low_findings": [],
                "total_findings": 0,
                "recommendations": [],
            }

        # Calculate weighted overall score
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        framework_scores: Dict[str, Dict[str, Any]] = {}

        for framework_name, check_result in results.items():
            # Look up weight (default 1.0)
            try:
                fw_enum = ComplianceFramework(framework_name)
                weight = FRAMEWORK_WEIGHTS.get(fw_enum, Decimal("1.00"))
            except ValueError:
                weight = Decimal("1.00")

            weighted_sum += check_result.score * weight
            total_weight += weight

            framework_scores[framework_name] = {
                "score": float(check_result.score),
                "status": check_result.status.value,
                "findings_count": len(check_result.findings),
                "weight": float(weight),
            }

        overall_score = Decimal("0")
        if total_weight > 0:
            overall_score = (weighted_sum / total_weight).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )

        # Determine overall status
        if overall_score >= Decimal("95"):
            overall_status = ComplianceStatus.PASS
        elif overall_score >= Decimal("70"):
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.FAIL

        # Aggregate findings by severity
        critical_findings: List[dict] = []
        high_findings: List[dict] = []
        medium_findings: List[dict] = []
        low_findings: List[dict] = []

        all_recommendations: List[str] = []
        seen_recommendations: Set[str] = set()

        for framework_name, check_result in results.items():
            for finding in check_result.findings:
                finding_with_fw = {
                    "framework": framework_name,
                    **finding,
                }
                severity = finding.get("severity", "")
                if severity == ComplianceSeverity.CRITICAL.value:
                    critical_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.HIGH.value:
                    high_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.MEDIUM.value:
                    medium_findings.append(finding_with_fw)
                elif severity == ComplianceSeverity.LOW.value:
                    low_findings.append(finding_with_fw)

                rec = finding.get("recommendation")
                if rec and rec not in seen_recommendations:
                    all_recommendations.append(rec)
                    seen_recommendations.add(rec)

        # Add priority recommendation if score is low
        if overall_score < Decimal("70"):
            priority_msg = (
                "PRIORITY: Overall compliance score is below 70%. "
                "Address critical and high-severity issues immediately."
            )
            if priority_msg not in seen_recommendations:
                all_recommendations.insert(0, priority_msg)

        total_findings = (
            len(critical_findings) + len(high_findings)
            + len(medium_findings) + len(low_findings)
        )

        return {
            "overall_score": float(overall_score),
            "overall_status": overall_status.value,
            "frameworks_checked": len(results),
            "framework_scores": framework_scores,
            "critical_findings": critical_findings,
            "high_findings": high_findings,
            "medium_findings": medium_findings,
            "low_findings": low_findings,
            "total_findings": total_findings,
            "recommendations": all_recommendations,
        }

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Return engine statistics.

        Returns:
            Dictionary with engine_id, version, check_count, and enabled frameworks.

        Example:
            >>> stats = engine.get_engine_stats()
            >>> stats["engine_id"]
            'compliance_checker_engine'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "check_count": self._check_count,
            "enabled_frameworks": self._enabled_frameworks,
            "strict_mode": self._config.compliance.strict_mode,
        }


# ==============================================================================
# MODULE-LEVEL ACCESSOR
# ==============================================================================


def get_compliance_checker() -> ComplianceCheckerEngine:
    """
    Get the ComplianceCheckerEngine singleton instance.

    Convenience function that delegates to the class-level get_instance().

    Returns:
        ComplianceCheckerEngine singleton.

    Example:
        >>> engine = get_compliance_checker()
        >>> engine.get_engine_stats()["engine_id"]
        'compliance_checker_engine'
    """
    return ComplianceCheckerEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "ComplianceSeverity",
    "DoubleCountingCategory",
    "BoundaryClassification",
    "ComplianceFinding",
    "FrameworkCheckState",
    "FRAMEWORK_WEIGHTS",
    "ComplianceCheckerEngine",
    "get_compliance_checker",
]
