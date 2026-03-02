# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-024 Engine 6

This module implements regulatory compliance checking for Use of Sold Products
emissions (GHG Protocol Scope 3 Category 11) against 7 regulatory frameworks
with ~50 framework rules and 8 double-counting prevention rules.

Category 11 is often the LARGEST Scope 3 category for manufacturers of
energy-consuming products (vehicles, appliances, electronics, HVAC).

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 11 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 Climate Change + E1-6
4. CDP Climate Change Questionnaire (C6.5 Category 11)
5. SBTi (Science Based Targets initiative, corporate net-zero)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 Emissions Standard

Category 11-Specific Compliance Rules:
- Direct vs indirect use-phase emission split disclosure
- Product lifetime documentation and assumptions
- Calculation method justification (direct fuel / direct refrigerant /
  direct chemical / indirect electricity / indirect heating / indirect
  steam-cooling / fuels sold / feedstocks sold)
- Use-profile assumptions documentation
- Double-counting prevention (8 rules: DC-USP-001 through DC-USP-008)
- Data quality scoring across product categories
- Materiality threshold assessment
- Boundary enforcement (Cat 11 vs Cat 10 vs Cat 12 vs Scope 1/2)

Double-Counting Prevention Rules:
    DC-USP-001: vs Scope 1 (own use of products - company uses own products)
    DC-USP-002: vs Scope 2 (own electricity from sold generators/panels)
    DC-USP-003: vs Cat 1 (upstream purchased goods - cradle-to-gate)
    DC-USP-004: vs Cat 3 (fuel & energy activities - WTT of fuels sold)
    DC-USP-005: vs Cat 10 (processing of sold products)
    DC-USP-006: vs Cat 12 (end-of-life of sold products)
    DC-USP-007: vs Cat 13 (downstream leased assets)
    DC-USP-008: Fuel double-count prevention (fuels sold as product vs input)

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all(calculation_result)
    >>> summary = engine.generate_report(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-011
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# Materiality threshold for SBTi (1% of total Scope 3)
SBTI_MATERIALITY_THRESHOLD: Decimal = Decimal("0.01")

# SB 253 reporting threshold (1% of total Scope 3)
SB253_REPORTING_THRESHOLD: Decimal = Decimal("0.01")

# Minimum data quality score for acceptable quality
MIN_DQI_SCORE: Decimal = Decimal("2.0")

# Maximum acceptable uncertainty percentage
MAX_UNCERTAINTY_PCT: Decimal = Decimal("50.0")


# ==============================================================================
# ENUMS
# ==============================================================================


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories that could overlap with Category 11."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    CATEGORY_1 = "CATEGORY_1"
    CATEGORY_3 = "CATEGORY_3"
    CATEGORY_10 = "CATEGORY_10"
    CATEGORY_12 = "CATEGORY_12"
    CATEGORY_13 = "CATEGORY_13"


class BoundaryClassification(str, Enum):
    """Category boundary classification for a product."""

    CATEGORY_11 = "CATEGORY_11"  # Use of sold products (correct)
    CATEGORY_10 = "CATEGORY_10"  # Processing of sold intermediates
    CATEGORY_12 = "CATEGORY_12"  # End-of-life treatment
    CATEGORY_13 = "CATEGORY_13"  # Downstream leased assets
    SCOPE_1 = "SCOPE_1"  # Own use of sold product
    SCOPE_2 = "SCOPE_2"  # Own electricity from sold product
    EXCLUDED = "EXCLUDED"  # Not in scope


class ProductUseCategory(str, Enum):
    """Product use categories for boundary checks."""

    VEHICLES = "vehicles"
    APPLIANCES = "appliances"
    HVAC = "hvac"
    LIGHTING = "lighting"
    IT_EQUIPMENT = "it_equipment"
    INDUSTRIAL_EQUIPMENT = "industrial_equipment"
    FUELS_FEEDSTOCKS = "fuels_feedstocks"
    BUILDING_PRODUCTS = "building_products"
    CONSUMER_PRODUCTS = "consumer_products"
    MEDICAL_DEVICES = "medical_devices"


class EmissionType(str, Enum):
    """Whether emissions are direct or indirect use-phase."""

    DIRECT = "direct"
    INDIRECT = "indirect"
    BOTH = "both"


class CalculationMethodUSP(str, Enum):
    """Calculation methods for Category 11."""

    DIRECT_FUEL_COMBUSTION = "direct_fuel_combustion"
    DIRECT_REFRIGERANT_LEAKAGE = "direct_refrigerant_leakage"
    DIRECT_CHEMICAL_RELEASE = "direct_chemical_release"
    INDIRECT_ELECTRICITY = "indirect_electricity"
    INDIRECT_HEATING_FUEL = "indirect_heating_fuel"
    INDIRECT_STEAM_COOLING = "indirect_steam_cooling"
    FUELS_SOLD = "fuels_sold"
    FEEDSTOCKS_SOLD = "feedstocks_sold"


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

        penalty_points = (
            Decimal(str(self.failed_checks))
            + Decimal(str(self.warning_checks)) * Decimal("0.5")
        )
        max_points = Decimal(str(self.total_checks))
        score = (
            (max_points - penalty_points) / max_points * Decimal("100")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

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

    def to_result(self) -> "ComplianceResult":
        """Convert accumulated state to a ComplianceResult."""
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

        return ComplianceResult(
            framework=self.framework,
            status=self.compute_status(),
            score=self.compute_score(),
            findings=findings_dicts,
            recommendations=recommendations,
            passed_checks=self.passed_checks,
            failed_checks=self.failed_checks,
            warning_checks=self.warning_checks,
            total_checks=self.total_checks,
        )


@dataclass
class ComplianceResult:
    """Result of compliance check for one framework."""

    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# SCORING WEIGHTS PER FRAMEWORK
# ==============================================================================

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
# REQUIRED DISCLOSURES PER FRAMEWORK
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_co2e",
        "direct_indirect_split",
        "product_categories",
        "lifetime_assumptions",
        "calculation_method",
        "ef_sources",
        "exclusions",
        "dqi_score",
    ],
    ComplianceFramework.ISO_14064: [
        "total_co2e",
        "uncertainty_analysis",
        "base_year",
        "methodology",
        "reporting_period",
        "verification_evidence",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_co2e",
        "methodology",
        "targets",
        "product_breakdown",
        "value_chain_boundary",
        "actions",
        "dnsh_assessment",
    ],
    ComplianceFramework.CDP: [
        "total_co2e",
        "product_breakdown",
        "data_quality",
        "methodology",
        "verification_status",
    ],
    ComplianceFramework.SBTI: [
        "total_co2e",
        "target_coverage",
        "progress_tracking",
        "base_year",
        "materiality",
        "reduction_pathway",
    ],
    ComplianceFramework.SB_253: [
        "total_co2e",
        "methodology",
        "assurance_opinion",
        "materiality",
        "reporting_threshold",
    ],
    ComplianceFramework.GRI: [
        "total_co2e",
        "gases_included",
        "base_year",
        "ef_sources",
    ],
}


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for Use of Sold Products emissions (Category 11).

    Validates calculation results against 7 regulatory frameworks with
    Category 11-specific rules including direct/indirect split disclosure,
    product lifetime assumptions, double-counting prevention, and
    category boundary enforcement.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _enabled_frameworks: Set of enabled compliance frameworks
        _check_count: Running count of compliance checks performed
        _strict_mode: If True, warnings become failures

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> results = engine.check_all(calc_result)
        >>> summary = engine.generate_report(results)
        >>> print(f"Overall: {summary['overall_status']}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with configuration."""
        self._enabled_frameworks: List[ComplianceFramework] = list(
            ComplianceFramework
        )
        self._check_count: int = 0
        self._strict_mode: bool = False
        self._materiality_threshold: Decimal = SBTI_MATERIALITY_THRESHOLD

        logger.info(
            "ComplianceCheckerEngine initialized: version=%s, "
            "frameworks=%d, strict_mode=%s",
            ENGINE_VERSION,
            len(self._enabled_frameworks),
            self._strict_mode,
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

    def check_all(
        self, result: dict
    ) -> Dict[str, ComplianceResult]:
        """
        Run all enabled framework checks and return results.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked.

        Args:
            result: Calculation result dictionary containing total_co2e,
                direct_co2e, indirect_co2e, product_categories,
                lifetime_assumptions, method, ef_sources, etc.

        Returns:
            Dictionary mapping framework name string to ComplianceResult.

        Example:
            >>> all_results = engine.check_all(calc_result)
            >>> ghg_result = all_results.get("ghg_protocol")
            >>> ghg_result.status
            <ComplianceStatus.PASS: 'PASS'>
        """
        start_time = time.monotonic()
        logger.info("Running compliance checks for all enabled frameworks")

        all_results: Dict[str, ComplianceResult] = {}

        framework_dispatch: Dict[ComplianceFramework, str] = {
            ComplianceFramework.GHG_PROTOCOL: "check_ghg_protocol",
            ComplianceFramework.ISO_14064: "check_iso_14064",
            ComplianceFramework.CSRD_ESRS: "check_csrd",
            ComplianceFramework.CDP: "check_cdp",
            ComplianceFramework.SBTI: "check_sbti",
            ComplianceFramework.SB_253: "check_sb253",
            ComplianceFramework.GRI: "check_gri",
        }

        for framework in self._enabled_frameworks:
            method_name = framework_dispatch.get(framework)
            if method_name is None:
                logger.warning(
                    "No dispatch method for framework '%s', skipping",
                    framework.value,
                )
                continue

            try:
                check_method = getattr(self, method_name)
                check_result = check_method(result)
                all_results[framework.value] = check_result

                logger.info(
                    "%s compliance: %s (score: %s)",
                    framework.value,
                    check_result.status.value,
                    check_result.score,
                )

            except Exception as e:
                logger.error(
                    "Error checking %s compliance: %s",
                    framework.value,
                    str(e),
                    exc_info=True,
                )
                all_results[framework.value] = ComplianceResult(
                    framework=framework,
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
    # Framework: GHG Protocol Scope 3 (Category 11)
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> ComplianceResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 11).

        Rules (~8):
            GHG-USP-001: Total CO2e present and positive
            GHG-USP-002: Direct vs indirect split documented
            GHG-USP-003: Product lifetime documented
            GHG-USP-004: Calculation method justified
            GHG-USP-005: Emission factor sources documented
            GHG-USP-006: Exclusions documented
            GHG-USP-007: DQI score present
            GHG-USP-008: Use-profile assumptions documented

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for GHG Protocol.

        Example:
            >>> res = engine.check_ghg_protocol(calc_result)
            >>> res.framework
            <ComplianceFramework.GHG_PROTOCOL: 'ghg_protocol'>
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # GHG-USP-001: Total emissions present and positive
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("GHG-USP-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-USP-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Ensure total_co2e is calculated and > 0 for Category 11. "
                    "Include both direct and indirect use-phase emissions."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11",
            )

        # GHG-USP-002: Direct vs indirect split documented
        direct_co2e = result.get("direct_co2e")
        indirect_co2e = result.get("indirect_co2e")
        emission_type = result.get("emission_type")
        has_split = (
            (direct_co2e is not None or indirect_co2e is not None)
            or emission_type is not None
        )
        if has_split:
            state.add_pass(
                "GHG-USP-002",
                "Direct vs indirect use-phase split documented",
            )
        else:
            state.add_fail(
                "GHG-USP-002",
                "Direct vs indirect use-phase split not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the split between direct use-phase emissions "
                    "(fuel combustion, refrigerant leakage, chemical release) "
                    "and indirect use-phase emissions (electricity, heating, "
                    "steam/cooling). GHG Protocol Ch 11 requires this distinction."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11, Table 11.1",
            )

        # GHG-USP-003: Product lifetime documented
        lifetime_years = result.get("lifetime_years")
        lifetime_assumptions = result.get("lifetime_assumptions")
        has_lifetime = lifetime_years is not None or lifetime_assumptions is not None
        if has_lifetime:
            state.add_pass("GHG-USP-003", "Product lifetime documented")
        else:
            state.add_fail(
                "GHG-USP-003",
                "Product lifetime assumptions not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the expected product lifetime in years, including "
                    "the basis for lifetime estimates (industry standards, "
                    "warranty periods, engineering estimates). Lifetime is a "
                    "critical parameter for Category 11 calculations."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11, Step 3",
            )

        # GHG-USP-004: Calculation method justified
        method = result.get("method") or result.get("calculation_method")
        if method:
            state.add_pass("GHG-USP-004", "Calculation method justified")
        else:
            state.add_fail(
                "GHG-USP-004",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used for Category 11 "
                    "(direct fuel combustion, direct refrigerant leakage, "
                    "direct chemical release, indirect electricity, "
                    "indirect heating, indirect steam/cooling, fuels sold, "
                    "or feedstocks sold)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11, Table 11.1",
            )

        # GHG-USP-005: Emission factor sources documented
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GHG-USP-005", "Emission factor sources documented")
        else:
            state.add_fail(
                "GHG-USP-005",
                "Emission factor sources not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document all emission factor sources used "
                    "(DEFRA, EPA, IEA, IPCC, custom). Include the publication "
                    "year and version of each factor set."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11",
            )

        # GHG-USP-006: Exclusions documented
        exclusions = result.get("exclusions")
        if exclusions is not None:
            state.add_pass("GHG-USP-006", "Exclusions documented")
        else:
            state.add_warning(
                "GHG-USP-006",
                "Exclusions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document any exclusions from Category 11 reporting "
                    "(e.g., products with de minimis use-phase emissions, "
                    "intermediate products processed by other companies)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11",
            )

        # GHG-USP-007: Data quality indicator score
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("GHG-USP-007", "DQI score present")
        else:
            state.add_warning(
                "GHG-USP-007",
                "Data quality indicator (DQI) score not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across 5 dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-USP-008: Use-profile assumptions documented
        use_profile = result.get("use_profile") or result.get("usage_assumptions")
        if use_profile is not None:
            state.add_pass(
                "GHG-USP-008",
                "Use-profile assumptions documented",
            )
        else:
            state.add_warning(
                "GHG-USP-008",
                "Use-profile assumptions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document assumptions about product use patterns, such as "
                    "hours of use per day, energy consumption rate, and usage "
                    "intensity. Use-profile assumptions significantly impact "
                    "Category 11 emission estimates."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 11, Step 2",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: ISO 14064
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> ComplianceResult:
        """
        Check compliance with ISO 14064-1:2018.

        Rules (~6):
            ISO-USP-001: Total CO2e present
            ISO-USP-002: Uncertainty analysis present
            ISO-USP-003: Base year documented
            ISO-USP-004: Methodology described
            ISO-USP-005: Reporting period defined
            ISO-USP-006: Verification evidence

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for ISO 14064.

        Example:
            >>> res = engine.check_iso_14064(calc_result)
            >>> res.framework.value
            'iso_14064'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14064)

        # ISO-USP-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("ISO-USP-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-USP-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total CO2e emissions.",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-USP-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-USP-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-USP-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis. Category 11 "
                    "typically has higher uncertainty due to lifetime and "
                    "use-profile assumptions. Consider Monte Carlo simulation "
                    "or IPCC Tier 2 default ranges."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-USP-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-USP-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-USP-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison. "
                    "Include rationale for base year selection and any "
                    "recalculation policy for product portfolio changes."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-USP-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-USP-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-USP-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, lifetime assumptions, use-profile "
                    "parameters, and calculation approach."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-USP-005: Reporting period defined
        reporting_period = (
            result.get("reporting_period")
            or result.get("period")
        )
        if reporting_period:
            state.add_pass("ISO-USP-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-USP-005",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Specify the reporting period (e.g., 2024, FY2024). "
                    "Category 11 reports on products sold in the reporting "
                    "period with lifetime emissions projected forward."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-USP-006: Verification evidence
        verification = (
            result.get("verification_evidence")
            or result.get("verification_status")
            or result.get("verified")
        )
        if verification is not None:
            state.add_pass("ISO-USP-006", "Verification evidence present")
        else:
            state.add_warning(
                "ISO-USP-006",
                "Verification evidence not provided",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide evidence of verification or state verification "
                    "status. ISO 14064-3 provides verification guidance."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 10",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CSRD / ESRS E1
    # ==========================================================================

    def check_csrd(self, result: dict) -> ComplianceResult:
        """
        Check compliance with CSRD ESRS E1 Climate Change.

        Rules (~7):
            CSRD-USP-001: Total CO2e by product category
            CSRD-USP-002: Methodology description
            CSRD-USP-003: Targets documented
            CSRD-USP-004: Product breakdown present
            CSRD-USP-005: Value chain boundary documented
            CSRD-USP-006: Actions described (transition plan)
            CSRD-USP-007: DNSH assessment (Do No Significant Harm)

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for CSRD/ESRS.

        Example:
            >>> res = engine.check_csrd(calc_result)
            >>> res.framework.value
            'csrd_esrs'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CSRD_ESRS)

        # CSRD-USP-001: Total emissions by product category
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("CSRD-USP-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-USP-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Scope 3 Category 11 emissions. "
                    "CSRD requires disclosure of significant Scope 3 "
                    "categories with disaggregated data."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-USP-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-USP-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-USP-002",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for use-of-sold "
                    "products emissions including direct/indirect split, "
                    "lifetime assumptions, and emission factors as required "
                    "by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-USP-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-USP-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-USP-003",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for Category 11 "
                    "(e.g., improve product energy efficiency, reduce "
                    "refrigerant charge, design for lower use-phase impact)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-USP-004: Product breakdown present
        product_breakdown = (
            result.get("product_breakdown")
            or result.get("by_category")
            or result.get("by_product")
        )
        if product_breakdown and isinstance(product_breakdown, dict) and len(product_breakdown) > 0:
            state.add_pass("CSRD-USP-004", "Product breakdown present")
        else:
            state.add_fail(
                "CSRD-USP-004",
                "Product breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by product category "
                    "(vehicles, appliances, HVAC, IT equipment, etc.). "
                    "CSRD requires disaggregated Scope 3 data."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-USP-005: Value chain boundary documented
        value_chain = (
            result.get("value_chain_boundary")
            or result.get("boundary")
        )
        if value_chain:
            state.add_pass("CSRD-USP-005", "Value chain boundary documented")
        else:
            state.add_warning(
                "CSRD-USP-005",
                "Value chain boundary not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the value chain boundary for Category 11, "
                    "clarifying what constitutes use-phase vs end-of-life "
                    "(Cat 12) vs processing of sold intermediates (Cat 10)."
                ),
                regulation_reference="ESRS E1-6, para 51(b)",
            )

        # CSRD-USP-006: Actions described (transition plan)
        actions = result.get("actions") or result.get("reduction_actions")
        if actions:
            state.add_pass("CSRD-USP-006", "Reduction actions described")
        else:
            state.add_warning(
                "CSRD-USP-006",
                "Reduction actions not described",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Describe actions taken or planned to reduce use-phase "
                    "emissions (e.g., improve energy efficiency, switch to "
                    "low-GWP refrigerants, design for circularity)."
                ),
                regulation_reference="ESRS E1-3, para 29",
            )

        # CSRD-USP-007: DNSH assessment
        dnsh = result.get("dnsh_assessment") or result.get("dnsh")
        if dnsh:
            state.add_pass("CSRD-USP-007", "DNSH assessment documented")
        else:
            state.add_warning(
                "CSRD-USP-007",
                "DNSH (Do No Significant Harm) assessment not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document a DNSH assessment per CSRD EU Taxonomy, "
                    "ensuring that product design changes to reduce Category "
                    "11 emissions do not significantly harm other "
                    "environmental objectives."
                ),
                regulation_reference="EU Taxonomy, Article 17",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CDP Climate Change
    # ==========================================================================

    def check_cdp(self, result: dict) -> ComplianceResult:
        """
        Check compliance with CDP Climate Change Questionnaire.

        Rules (~5):
            CDP-USP-001: Total CO2e present (C6.5)
            CDP-USP-002: Product breakdown by category
            CDP-USP-003: Data quality assessment
            CDP-USP-004: Methodology documented
            CDP-USP-005: Verification status

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for CDP.

        Example:
            >>> res = engine.check_cdp(calc_result)
            >>> res.framework.value
            'cdp'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CDP)

        # CDP-USP-001: Total emissions present (C6.5 Category 11)
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("CDP-USP-001", "Total CO2e reported for C6.5")
        else:
            state.add_fail(
                "CDP-USP-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 11 emissions to CDP C6.5. "
                    "Include both direct and indirect use-phase emissions."
                ),
                regulation_reference="CDP CC Module C6.5, Category 11",
            )

        # CDP-USP-002: Product breakdown by category
        product_breakdown = (
            result.get("product_breakdown")
            or result.get("by_category")
            or result.get("by_product")
        )
        if product_breakdown and isinstance(product_breakdown, dict) and len(product_breakdown) > 0:
            state.add_pass("CDP-USP-002", "Product breakdown present for CDP")
        else:
            state.add_fail(
                "CDP-USP-002",
                "Product breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by product category. "
                    "CDP requires granular Category 11 disclosure."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-USP-003: Data quality assessment
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        data_quality = result.get("data_quality")
        if dqi_score is not None or data_quality is not None:
            state.add_pass("CDP-USP-003", "Data quality assessment present")
        else:
            state.add_warning(
                "CDP-USP-003",
                "Data quality assessment not provided",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide data quality assessment for Category 11. "
                    "CDP scores data quality in its rating methodology."
                ),
                regulation_reference="CDP CC Scoring Methodology",
            )

        # CDP-USP-004: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
        )
        if methodology:
            state.add_pass("CDP-USP-004", "Methodology documented for CDP")
        else:
            state.add_warning(
                "CDP-USP-004",
                "Methodology not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the methodology used for Category 11, "
                    "including emission factor sources and assumptions."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-USP-005: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-USP-005", "Verification status documented")
        else:
            state.add_warning(
                "CDP-USP-005",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 11 emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module C10.1",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SBTi
    # ==========================================================================

    def check_sbti(self, result: dict) -> ComplianceResult:
        """
        Check compliance with Science Based Targets initiative.

        Rules (~6):
            SBTI-USP-001: Total CO2e present
            SBTI-USP-002: Target coverage documented (67% minimum)
            SBTI-USP-003: Base year documented
            SBTI-USP-004: Progress tracking present
            SBTI-USP-005: Materiality assessment
            SBTI-USP-006: Reduction pathway documented

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for SBTi.

        Example:
            >>> res = engine.check_sbti(calc_result)
            >>> res.framework.value
            'sbti'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SBTI)

        # SBTI-USP-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("SBTI-USP-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-USP-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Calculate total Category 11 emissions. For many "
                    "manufacturers, Category 11 is the largest Scope 3 "
                    "category and critical for SBTi target boundary."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-USP-002: Target coverage documented
        target_coverage = (
            result.get("target_coverage")
            or result.get("sbti_coverage")
        )
        if target_coverage is not None:
            try:
                coverage_val = self._to_decimal(target_coverage)
                if coverage_val >= Decimal("67"):
                    state.add_pass(
                        "SBTI-USP-002",
                        f"Target coverage {coverage_val}% meets SBTi minimum 67%",
                    )
                else:
                    state.add_warning(
                        "SBTI-USP-002",
                        f"Target coverage {coverage_val}% below SBTi minimum 67%",
                        ComplianceSeverity.HIGH,
                        recommendation=(
                            "SBTi requires minimum 67% Scope 3 coverage. "
                            "Category 11 is often key to meeting this threshold."
                        ),
                        regulation_reference="SBTi Criteria v5.1, C20",
                    )
            except (InvalidOperation, ValueError):
                state.add_pass(
                    "SBTI-USP-002",
                    "Target coverage documented (value check skipped)",
                )
        else:
            state.add_warning(
                "SBTI-USP-002",
                "SBTi target coverage not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document what percentage of Scope 3 emissions is covered "
                    "by SBTi targets (minimum 67% required)."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-USP-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("SBTI-USP-003", "Base year documented for SBTi")
        else:
            state.add_fail(
                "SBTI-USP-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for SBTi target setting. "
                    "The base year should reflect a representative year "
                    "for the product portfolio."
                ),
                regulation_reference="SBTi Criteria v5.1, C5",
            )

        # SBTI-USP-004: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-USP-004", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-USP-004",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 11 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-USP-005: Materiality assessment
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat11_pct = (
                    self._to_decimal(total_co2e)
                    / self._to_decimal(total_scope3)
                    * Decimal("100")
                )
                if cat11_pct >= self._materiality_threshold * Decimal("100"):
                    state.add_pass(
                        "SBTI-USP-005",
                        f"Category 11 is material ({cat11_pct:.1f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SBTI-USP-005",
                        f"Category 11 below materiality threshold "
                        f"({cat11_pct:.1f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 11 may not need to be included in SBTi "
                            "target boundary if below materiality threshold."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-USP-005",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-USP-005",
                "Total Scope 3 emissions not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 11 "
                    "materiality. For manufacturers, Cat 11 is often >40% "
                    "of total Scope 3."
                ),
            )

        # SBTI-USP-006: Reduction pathway documented
        reduction_pathway = (
            result.get("reduction_pathway")
            or result.get("decarbonization_levers")
        )
        if reduction_pathway:
            state.add_pass("SBTI-USP-006", "Reduction pathway documented")
        else:
            state.add_warning(
                "SBTI-USP-006",
                "Reduction pathway not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the decarbonization levers for Category 11 "
                    "(e.g., improve product energy efficiency, transition "
                    "to lower-carbon energy sources, reduce refrigerant GWP, "
                    "extend product lifetime)."
                ),
                regulation_reference="SBTi Net-Zero Standard, v1.0",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SB 253
    # ==========================================================================

    def check_sb253(self, result: dict) -> ComplianceResult:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Rules (~5):
            SB253-USP-001: Total CO2e present
            SB253-USP-002: Methodology documented
            SB253-USP-003: Assurance opinion available
            SB253-USP-004: Reporting threshold check
            SB253-USP-005: Materiality > 1% threshold

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for SB 253.

        Example:
            >>> res = engine.check_sb253(calc_result)
            >>> res.framework.value
            'sb_253'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # SB253-USP-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("SB253-USP-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-USP-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 11 emissions for SB 253 compliance."
                ),
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-USP-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-USP-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-USP-002",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-USP-003: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-USP-003", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-USP-003",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-USP-004: Reporting threshold check
        total_scope3 = result.get("total_scope3_co2e")
        reporting_threshold = result.get("reporting_threshold")
        if reporting_threshold is not None:
            state.add_pass(
                "SB253-USP-004",
                "Reporting threshold documented",
            )
        elif total_scope3 is not None:
            state.add_pass(
                "SB253-USP-004",
                "Total Scope 3 available for threshold evaluation",
            )
        else:
            state.add_warning(
                "SB253-USP-004",
                "Reporting threshold not evaluated",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Evaluate whether Category 11 exceeds the SB 253 "
                    "reporting threshold. Include total Scope 3 for context."
                ),
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-USP-005: Materiality > 1% threshold
        if total_co2e and total_scope3:
            try:
                cat11_pct = (
                    self._to_decimal(total_co2e)
                    / self._to_decimal(total_scope3)
                    * Decimal("100")
                )
                if cat11_pct > Decimal("1"):
                    state.add_pass(
                        "SB253-USP-005",
                        f"Category 11 exceeds 1% materiality "
                        f"({cat11_pct:.2f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SB253-USP-005",
                        f"Category 11 below 1% materiality threshold "
                        f"({cat11_pct:.2f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 11 is below 1% of total Scope 3. "
                            "Consider whether separate reporting is warranted."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SB253-USP-005",
                    "Could not calculate materiality",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SB253-USP-005",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 11 materiality against 1% threshold."
                ),
            )

        return state.to_result()

    # ==========================================================================
    # Framework: GRI 305
    # ==========================================================================

    def check_gri(self, result: dict) -> ComplianceResult:
        """
        Check compliance with GRI 305 Emissions Standard.

        Rules (~4):
            GRI-USP-001: Total CO2e in metric tonnes
            GRI-USP-002: Gases included documented
            GRI-USP-003: Base year present
            GRI-USP-004: Emission factor sources documented

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceResult for GRI.

        Example:
            >>> res = engine.check_gri(calc_result)
            >>> res.framework.value
            'gri'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GRI)

        # GRI-USP-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and self._to_decimal(total_co2e) > 0:
            state.add_pass("GRI-USP-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-USP-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 11 emissions in metric tonnes CO2e."
                ),
                regulation_reference="GRI 305-3",
            )

        # GRI-USP-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-USP-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-USP-002",
                "Gases included not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, HFCs, PFCs, SF6, NF3). Category 11 "
                    "may include refrigerants (HFCs) for HVAC products."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-USP-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-USP-003", "Base year present")
        else:
            state.add_warning(
                "GRI-USP-003",
                "Base year not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-USP-004: Emission factor sources documented
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-USP-004", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-USP-004",
                "Emission factor sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., DEFRA 2024, EPA eGRID 2024, IEA 2024)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention (8 Rules)
    # ==========================================================================

    def check_double_counting(
        self, products: list
    ) -> List[Dict[str, Any]]:
        """
        Validate 8 double-counting prevention rules for use of sold products.

        Rules:
            DC-USP-001: vs Scope 1 (company uses its own sold products)
            DC-USP-002: vs Scope 2 (own electricity from sold generators/panels)
            DC-USP-003: vs Cat 1 (upstream purchased goods, cradle-to-gate)
            DC-USP-004: vs Cat 3 (fuel & energy WTT of fuels sold)
            DC-USP-005: vs Cat 10 (processing of sold intermediate products)
            DC-USP-006: vs Cat 12 (end-of-life of sold products)
            DC-USP-007: vs Cat 13 (downstream leased assets)
            DC-USP-008: Fuel double-count prevention

        Args:
            products: List of product dictionaries, each containing fields
                like product_category, sold_to_self, generates_electricity,
                is_intermediate, reported_in_cat10/12/13, fuel_type, etc.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and affected product indices.

        Example:
            >>> findings = engine.check_double_counting(products)
            >>> len(findings)
            0  # No double-counting issues
        """
        findings: List[Dict[str, Any]] = []

        for idx, product in enumerate(products):
            findings.extend(self._check_dc_usp_001(idx, product))
            findings.extend(self._check_dc_usp_002(idx, product))
            findings.extend(self._check_dc_usp_003(idx, product))
            findings.extend(self._check_dc_usp_004(idx, product))
            findings.extend(self._check_dc_usp_005(idx, product))
            findings.extend(self._check_dc_usp_006(idx, product))
            findings.extend(self._check_dc_usp_007(idx, product))
            findings.extend(self._check_dc_usp_008(idx, product))

        logger.info(
            "Double-counting check: %d products, %d findings",
            len(products),
            len(findings),
        )

        return findings

    def _check_dc_usp_001(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-001: vs Scope 1 (own use of products).

        If the reporting company uses its own sold products, those
        emissions should be in Scope 1, not Category 11.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        sold_to_self = product.get("sold_to_self", False)
        own_use = product.get("own_use", False)
        if sold_to_self or own_use:
            findings.append({
                "rule_code": "DC-USP-001",
                "description": (
                    f"Product {idx}: Sold product is used by the reporting "
                    "company itself. These emissions should be reported "
                    "under Scope 1, not Category 11."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "product_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "Exclude products consumed by the reporting company "
                    "from Category 11. Report their emissions under "
                    "Scope 1 (direct emissions)."
                ),
            })

        return findings

    def _check_dc_usp_002(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-002: vs Scope 2 (own electricity from sold products).

        If the reporting company generates electricity from its own
        sold generators/solar panels, those emissions should be in
        Scope 2, not Category 11.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        generates_electricity = product.get("generates_electricity_for_self", False)
        own_energy = product.get("own_energy_source", False)
        if generates_electricity or own_energy:
            findings.append({
                "rule_code": "DC-USP-002",
                "description": (
                    f"Product {idx}: Sold product generates electricity "
                    "consumed by the reporting company. These emissions "
                    "should be Scope 2, not Category 11."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "product_index": idx,
                "category": DoubleCountingCategory.SCOPE_2.value,
                "recommendation": (
                    "Exclude self-consumed electricity generation from "
                    "Category 11. Report under Scope 2 (purchased electricity)."
                ),
            })

        return findings

    def _check_dc_usp_003(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-003: vs Cat 1 (upstream purchased goods).

        Category 1 covers cradle-to-gate emissions. Category 11 covers
        use-phase. There should be no overlap in the system boundary.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        includes_upstream = product.get("includes_upstream_emissions", False)
        reported_in_cat1 = product.get("reported_in_cat1", False)
        if includes_upstream and reported_in_cat1:
            findings.append({
                "rule_code": "DC-USP-003",
                "description": (
                    f"Product {idx}: Category 11 calculation includes "
                    "upstream (cradle-to-gate) emissions that are already "
                    "reported in Category 1 (Purchased Goods & Services)."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "product_index": idx,
                "category": DoubleCountingCategory.CATEGORY_1.value,
                "recommendation": (
                    "Ensure Category 11 boundary starts at the point of "
                    "sale (use-phase only). Cradle-to-gate emissions belong "
                    "in Category 1. Do not include manufacturing or "
                    "transportation emissions in Category 11."
                ),
            })

        return findings

    def _check_dc_usp_004(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-004: vs Cat 3 (fuel & energy WTT of fuels sold).

        If the company sells fuels, the WTT (well-to-tank) emissions
        for those fuels may be in Category 3, while the combustion
        emissions are in Category 11. These must not overlap.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        is_fuel = product.get("is_fuel_product", False)
        wtt_included = product.get("wtt_included_in_cat11", False)
        wtt_in_cat3 = product.get("wtt_reported_in_cat3", False)
        if is_fuel and wtt_included and wtt_in_cat3:
            findings.append({
                "rule_code": "DC-USP-004",
                "description": (
                    f"Product {idx}: WTT emissions for sold fuel are "
                    "included in both Category 11 and Category 3 (Fuel & "
                    "Energy Activities). Double-counting risk."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "product_index": idx,
                "category": DoubleCountingCategory.CATEGORY_3.value,
                "recommendation": (
                    "Report WTT emissions for sold fuels in either "
                    "Category 3 or Category 11, not both. Typically WTT "
                    "belongs in Category 3 while combustion belongs in "
                    "Category 11."
                ),
            })

        return findings

    def _check_dc_usp_005(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-005: vs Cat 10 (processing of sold products).

        If a sold product is an intermediate product processed by
        another company, the processing emissions belong in Category 10,
        not Category 11.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        is_intermediate = product.get("is_intermediate", False)
        reported_in_cat10 = product.get("reported_in_cat10", False)
        processing_included = product.get("includes_processing", False)
        if is_intermediate and (reported_in_cat10 or processing_included):
            findings.append({
                "rule_code": "DC-USP-005",
                "description": (
                    f"Product {idx}: Intermediate product with processing "
                    "emissions. Processing emissions should be in Category "
                    "10, not Category 11. Category 11 is for end-use "
                    "emissions only."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "product_index": idx,
                "category": DoubleCountingCategory.CATEGORY_10.value,
                "recommendation": (
                    "Report processing emissions for intermediate products "
                    "in Category 10. Category 11 should only include "
                    "emissions from the final consumer use phase."
                ),
            })

        return findings

    def _check_dc_usp_006(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-006: vs Cat 12 (end-of-life of sold products).

        End-of-life treatment emissions belong in Category 12.
        Category 11 should only include use-phase emissions and must
        not overlap with end-of-life disposal.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        includes_eol = product.get("includes_end_of_life", False)
        reported_in_cat12 = product.get("reported_in_cat12", False)
        if includes_eol:
            findings.append({
                "rule_code": "DC-USP-006",
                "description": (
                    f"Product {idx}: Category 11 calculation includes "
                    "end-of-life emissions. End-of-life treatment belongs "
                    "in Category 12, not Category 11."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "product_index": idx,
                "category": DoubleCountingCategory.CATEGORY_12.value,
                "recommendation": (
                    "Remove end-of-life emissions from Category 11. "
                    "Report disposal, recycling, and treatment emissions "
                    "in Category 12 (End-of-Life Treatment)."
                ),
            })
        elif reported_in_cat12:
            # This is correct boundary enforcement - no finding needed
            pass

        return findings

    def _check_dc_usp_007(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-007: vs Cat 13 (downstream leased assets).

        If a sold product is also leased downstream, the lease emissions
        should be in Category 13, not Category 11.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        is_leased = product.get("is_leased_downstream", False)
        reported_in_cat13 = product.get("reported_in_cat13", False)
        also_in_cat11 = product.get("also_in_cat11", True)
        if is_leased and also_in_cat11 and reported_in_cat13:
            findings.append({
                "rule_code": "DC-USP-007",
                "description": (
                    f"Product {idx}: Product is leased downstream and "
                    "reported in both Category 11 and Category 13. "
                    "Double-counting risk."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "product_index": idx,
                "category": DoubleCountingCategory.CATEGORY_13.value,
                "recommendation": (
                    "Leased assets should be reported in either Category 11 "
                    "(use of sold products) or Category 13 (downstream "
                    "leased assets), not both. Typically, leased-out assets "
                    "go in Category 13."
                ),
            })

        return findings

    def _check_dc_usp_008(
        self, idx: int, product: dict
    ) -> List[Dict[str, Any]]:
        """
        DC-USP-008: Fuel double-count prevention.

        When a company sells fuels, the combustion emissions are in
        Category 11 for the fuel seller. However, these same emissions
        are reported as Scope 1 by the fuel purchaser/user. This is
        acceptable cross-company but must not be double-counted within
        the same reporting entity.

        Args:
            idx: Product index.
            product: Product dictionary.

        Returns:
            List of findings (empty if no issue).
        """
        findings: List[Dict[str, Any]] = []

        is_fuel = product.get("is_fuel_product", False)
        fuel_used_internally = product.get("fuel_used_internally", False)
        if is_fuel and fuel_used_internally:
            findings.append({
                "rule_code": "DC-USP-008",
                "description": (
                    f"Product {idx}: Sold fuel is also used internally by "
                    "the reporting company. Internal fuel consumption should "
                    "be Scope 1 (stationary/mobile combustion), not "
                    "Category 11."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "product_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "Exclude internally consumed fuel quantities from "
                    "Category 11 totals. Only fuel sold to external "
                    "customers should be reported as Category 11. "
                    "Internal consumption belongs in Scope 1."
                ),
            })

        return findings

    # ==========================================================================
    # Boundary Validation
    # ==========================================================================

    def validate_boundary(
        self, product: dict
    ) -> Dict[str, Any]:
        """
        Determine if a product belongs in Category 11, Cat 10, Cat 12,
        Cat 13, Scope 1, or is excluded.

        Classification Rules:
            - Own use of product -> Scope 1
            - Own electricity from product -> Scope 2
            - Intermediate product for processing -> Category 10
            - End-of-life treatment -> Category 12
            - Downstream leased asset -> Category 13
            - External customer use-phase -> Category 11

        Args:
            product: Product dictionary with category, sold_to, ownership
                and usage context fields.

        Returns:
            Dictionary with classification, reason, and any warnings.

        Example:
            >>> result = engine.validate_boundary({
            ...     "product_category": "vehicles",
            ...     "sold_to": "external_customer",
            ... })
            >>> result["classification"]
            'CATEGORY_11'
        """
        warnings: List[str] = []

        # Rule 1: Own use -> Scope 1
        sold_to_self = product.get("sold_to_self", False)
        own_use = product.get("own_use", False)
        if sold_to_self or own_use:
            return {
                "classification": BoundaryClassification.SCOPE_1.value,
                "reason": (
                    "Product used by the reporting company. Use-phase "
                    "emissions belong in Scope 1, not Category 11."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 2: Own electricity -> Scope 2
        generates_electricity = product.get("generates_electricity_for_self", False)
        if generates_electricity:
            return {
                "classification": BoundaryClassification.SCOPE_2.value,
                "reason": (
                    "Product generates electricity consumed by reporting "
                    "company. Belongs in Scope 2."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 3: Intermediate product -> Category 10
        is_intermediate = product.get("is_intermediate", False)
        if is_intermediate:
            return {
                "classification": BoundaryClassification.CATEGORY_10.value,
                "reason": (
                    "Intermediate product for further processing. "
                    "Processing emissions belong in Category 10."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 4: Downstream leased -> Category 13
        is_leased = product.get("is_leased_downstream", False)
        if is_leased:
            return {
                "classification": BoundaryClassification.CATEGORY_13.value,
                "reason": (
                    "Product is leased downstream. Lease-related "
                    "use-phase emissions belong in Category 13."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 5: End-of-life treatment context -> Category 12
        is_eol = product.get("is_end_of_life_context", False)
        if is_eol:
            return {
                "classification": BoundaryClassification.CATEGORY_12.value,
                "reason": (
                    "Product in end-of-life context. Disposal/treatment "
                    "emissions belong in Category 12."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 6: External customer use -> Category 11
        product_category = product.get("product_category", "")
        if product_category:
            # Check for known use categories
            valid_categories = {c.value for c in ProductUseCategory}
            if product_category.lower() not in valid_categories:
                warnings.append(
                    f"Unknown product category '{product_category}'. "
                    "Verify it belongs in Category 11."
                )

        return {
            "classification": BoundaryClassification.CATEGORY_11.value,
            "reason": (
                "Product sold to external customers for use. "
                "Use-phase emissions belong in Category 11."
            ),
            "warnings": warnings,
            "product": product,
        }

    # ==========================================================================
    # Completeness Validation
    # ==========================================================================

    def validate_completeness(
        self, result: dict
    ) -> Dict[str, Any]:
        """
        Validate completeness of a Category 11 calculation result.

        Checks for missing required fields, partial data, and
        coverage gaps across product categories.

        Args:
            result: Calculation result dictionary.

        Returns:
            Dictionary with completeness_score (0-100),
            missing_fields, and recommendations.

        Example:
            >>> completeness = engine.validate_completeness(calc_result)
            >>> completeness["completeness_score"]
            85.0
        """
        required_fields = [
            "total_co2e",
            "direct_co2e",
            "indirect_co2e",
            "method",
            "ef_sources",
            "product_categories",
            "lifetime_assumptions",
            "units_sold",
        ]

        present_fields: List[str] = []
        missing_fields: List[str] = []
        recommendations: List[str] = []

        for field_name in required_fields:
            value = result.get(field_name)
            if value is not None and value != "" and value != []:
                present_fields.append(field_name)
            else:
                missing_fields.append(field_name)

        completeness_score = (
            len(present_fields) / len(required_fields) * 100.0
            if required_fields
            else 0.0
        )

        if "total_co2e" in missing_fields:
            recommendations.append(
                "Calculate total CO2e emissions for all sold products."
            )
        if "direct_co2e" in missing_fields:
            recommendations.append(
                "Separate direct use-phase emissions (fuel, refrigerant, chemical)."
            )
        if "indirect_co2e" in missing_fields:
            recommendations.append(
                "Separate indirect use-phase emissions (electricity, heating, steam)."
            )
        if "lifetime_assumptions" in missing_fields:
            recommendations.append(
                "Document product lifetime assumptions for each product category."
            )
        if "units_sold" in missing_fields:
            recommendations.append(
                "Provide units sold per product category for the reporting period."
            )
        if "product_categories" in missing_fields:
            recommendations.append(
                "List all product categories included in the calculation."
            )

        return {
            "completeness_score": round(completeness_score, 1),
            "present_fields": present_fields,
            "missing_fields": missing_fields,
            "total_required": len(required_fields),
            "total_present": len(present_fields),
            "recommendations": recommendations,
        }

    # ==========================================================================
    # Lifetime Assumptions Validation
    # ==========================================================================

    def validate_lifetime_assumptions(
        self, assumptions: dict
    ) -> Dict[str, Any]:
        """
        Validate product lifetime assumptions used in Category 11.

        Checks that lifetime values are within reasonable ranges for
        each product category and that the source of lifetime estimates
        is documented.

        Args:
            assumptions: Dictionary mapping product_category to
                lifetime configuration dict with 'years', 'source',
                'degradation_rate', etc.

        Returns:
            Dictionary with validation results per category.

        Example:
            >>> result = engine.validate_lifetime_assumptions({
            ...     "vehicles": {"years": 15, "source": "industry_average"},
            ...     "appliances": {"years": 12, "source": "warranty_period"},
            ... })
            >>> result["all_valid"]
            True
        """
        # Reasonable lifetime ranges by category (years)
        expected_ranges: Dict[str, Tuple[int, int]] = {
            "vehicles": (8, 25),
            "appliances": (5, 20),
            "hvac": (10, 25),
            "lighting": (1, 15),
            "it_equipment": (3, 10),
            "industrial_equipment": (10, 40),
            "fuels_feedstocks": (0, 1),  # Consumed immediately
            "building_products": (15, 50),
            "consumer_products": (0, 5),
            "medical_devices": (5, 20),
        }

        results: Dict[str, Dict[str, Any]] = {}
        all_valid = True
        issues: List[str] = []

        for category, config in assumptions.items():
            category_lower = category.lower()
            lifetime_years = config.get("years")
            source = config.get("source")
            degradation_rate = config.get("degradation_rate")

            category_result: Dict[str, Any] = {
                "valid": True,
                "warnings": [],
                "errors": [],
            }

            # Check lifetime years
            if lifetime_years is None:
                category_result["valid"] = False
                category_result["errors"].append(
                    "Lifetime years not specified."
                )
                all_valid = False
            else:
                try:
                    years = int(lifetime_years)
                    expected = expected_ranges.get(category_lower)
                    if expected:
                        min_yrs, max_yrs = expected
                        if years < min_yrs:
                            category_result["warnings"].append(
                                f"Lifetime {years} years is below expected "
                                f"minimum ({min_yrs}) for {category_lower}."
                            )
                        elif years > max_yrs:
                            category_result["warnings"].append(
                                f"Lifetime {years} years exceeds expected "
                                f"maximum ({max_yrs}) for {category_lower}."
                            )
                    if years <= 0:
                        category_result["valid"] = False
                        category_result["errors"].append(
                            "Lifetime years must be positive."
                        )
                        all_valid = False
                except (ValueError, TypeError):
                    category_result["valid"] = False
                    category_result["errors"].append(
                        f"Invalid lifetime value: {lifetime_years}"
                    )
                    all_valid = False

            # Check source documentation
            if not source:
                category_result["warnings"].append(
                    "Lifetime source not documented. Document the basis "
                    "(industry standard, warranty, engineering estimate)."
                )

            # Check degradation rate (optional but recommended)
            if degradation_rate is not None:
                try:
                    rate = float(degradation_rate)
                    if rate < 0 or rate > 1:
                        category_result["warnings"].append(
                            f"Degradation rate {rate} outside 0-1 range."
                        )
                except (ValueError, TypeError):
                    category_result["warnings"].append(
                        f"Invalid degradation rate: {degradation_rate}"
                    )

            if category_result["warnings"]:
                issues.extend(category_result["warnings"])

            results[category] = category_result

        return {
            "all_valid": all_valid,
            "category_results": results,
            "issues": issues,
            "total_categories": len(assumptions),
        }

    # ==========================================================================
    # Report Generation
    # ==========================================================================

    def generate_report(
        self, results: Dict[str, ComplianceResult]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive compliance report from all framework results.

        Calculates a weighted average compliance score across all checked
        frameworks and aggregates findings by severity.

        Args:
            results: Dictionary mapping framework name to ComplianceResult.

        Returns:
            Report dictionary with overall_score, overall_status,
            framework_scores, findings by severity, and recommendations.

        Example:
            >>> report = engine.generate_report(all_results)
            >>> report["overall_score"]
            85.5
            >>> report["overall_status"]
            'WARNING'
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
                "info_findings": [],
                "total_findings": 0,
                "recommendations": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "engine_version": ENGINE_VERSION,
            }

        # Calculate weighted overall score
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")
        framework_scores: Dict[str, Dict[str, Any]] = {}

        for framework_name, check_result in results.items():
            try:
                framework_enum = ComplianceFramework(framework_name)
                weight = FRAMEWORK_WEIGHTS.get(
                    framework_enum, Decimal("1.00")
                )
            except ValueError:
                weight = Decimal("1.00")

            weighted_sum += check_result.score * weight
            total_weight += weight

            framework_scores[framework_name] = {
                "score": float(check_result.score),
                "status": check_result.status.value,
                "passed": check_result.passed_checks,
                "failed": check_result.failed_checks,
                "warnings": check_result.warning_checks,
                "total": check_result.total_checks,
                "weight": float(weight),
            }

        overall_score = Decimal("0.00")
        if total_weight > 0:
            overall_score = (weighted_sum / total_weight).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )

        # Determine overall status
        has_critical_fail = any(
            r.status == ComplianceStatus.FAIL for r in results.values()
        )
        has_warning = any(
            r.status == ComplianceStatus.WARNING for r in results.values()
        )

        if has_critical_fail:
            overall_status = ComplianceStatus.FAIL
        elif has_warning:
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.PASS

        # Aggregate findings by severity
        critical_findings: List[Dict[str, Any]] = []
        high_findings: List[Dict[str, Any]] = []
        medium_findings: List[Dict[str, Any]] = []
        low_findings: List[Dict[str, Any]] = []
        info_findings: List[Dict[str, Any]] = []
        all_recommendations: List[str] = []

        for framework_name, check_result in results.items():
            for finding in check_result.findings:
                severity = finding.get("severity", "INFO")
                tagged_finding = {**finding, "framework": framework_name}

                if severity == ComplianceSeverity.CRITICAL.value:
                    critical_findings.append(tagged_finding)
                elif severity == ComplianceSeverity.HIGH.value:
                    high_findings.append(tagged_finding)
                elif severity == ComplianceSeverity.MEDIUM.value:
                    medium_findings.append(tagged_finding)
                elif severity == ComplianceSeverity.LOW.value:
                    low_findings.append(tagged_finding)
                else:
                    info_findings.append(tagged_finding)

            all_recommendations.extend(check_result.recommendations)

        # Deduplicate recommendations
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        total_findings = (
            len(critical_findings)
            + len(high_findings)
            + len(medium_findings)
            + len(low_findings)
            + len(info_findings)
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
            "info_findings": info_findings,
            "total_findings": total_findings,
            "recommendations": unique_recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
        }

    # ==========================================================================
    # Configuration Methods
    # ==========================================================================

    def set_strict_mode(self, strict: bool) -> None:
        """
        Set strict mode where warnings become failures.

        Args:
            strict: True to enable strict mode.
        """
        self._strict_mode = strict
        logger.info("Strict mode set to %s", strict)

    def set_enabled_frameworks(
        self, frameworks: List[ComplianceFramework]
    ) -> None:
        """
        Set which frameworks are enabled for checking.

        Args:
            frameworks: List of ComplianceFramework enums to enable.
        """
        self._enabled_frameworks = frameworks
        logger.info(
            "Enabled frameworks updated: %s",
            [f.value for f in frameworks],
        )

    def get_enabled_frameworks(self) -> List[str]:
        """
        Get the list of currently enabled frameworks.

        Returns:
            List of framework name strings.
        """
        return [f.value for f in self._enabled_frameworks]

    def get_check_count(self) -> int:
        """
        Get the total number of compliance checks performed.

        Returns:
            Integer count of check_all invocations.
        """
        return self._check_count

    def get_required_disclosures(
        self, framework: ComplianceFramework
    ) -> List[str]:
        """
        Get the list of required disclosures for a framework.

        Args:
            framework: ComplianceFramework enum.

        Returns:
            List of required field names.
        """
        return FRAMEWORK_REQUIRED_DISCLOSURES.get(framework, [])

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and configuration.

        Returns:
            Dictionary with engine id, version, status, and config.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "enabled_frameworks": self.get_enabled_frameworks(),
            "strict_mode": self._strict_mode,
            "check_count": self._check_count,
            "materiality_threshold": float(self._materiality_threshold),
            "framework_weights": {
                k.value: float(v)
                for k, v in FRAMEWORK_WEIGHTS.items()
            },
        }

    # ==========================================================================
    # Private Helpers
    # ==========================================================================

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        """
        Safely convert a value to Decimal.

        Args:
            value: Value to convert (int, float, str, Decimal).

        Returns:
            Decimal representation.

        Raises:
            InvalidOperation: If value cannot be converted.
        """
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
