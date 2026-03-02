# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-025 Engine 6

This module implements regulatory compliance checking for End-of-Life Treatment
of Sold Products emissions (GHG Protocol Scope 3 Category 12) against 7 regulatory
frameworks with Category 12-specific compliance rules.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 12 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 + E5 (Climate Change + Resource Use & Circular Economy)
4. CDP Climate Change Questionnaire (C6.5 Category 12)
5. SBTi (Science Based Targets initiative)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 + 306 Emissions & Waste Standards

Category 12-Specific Compliance Rules:
- Product boundary enforcement: only products SOLD by reporting company
- 3 calculation methods hierarchy (waste-type-specific / average-data / producer-specific)
- Double-counting prevention (8 rules: DC-EOL-001 through DC-EOL-008)
- Material composition and treatment method breakdown
- Avoided emissions from recycling reported SEPARATELY (not netted off)
- Energy recovery credits reported SEPARATELY (not netted off)
- Biogenic vs fossil CO2 separation
- Circularity metrics (recycling rate, diversion rate, waste hierarchy)
- Data quality scoring
- Completeness scoring across product portfolio

Double-Counting Prevention Rules:
    DC-EOL-001: vs Cat 5 (own waste != customer disposal)
    DC-EOL-002: vs Cat 1 (upstream cradle-to-gate excludes post-consumer EOL)
    DC-EOL-003: vs Cat 11 (use-phase != end-of-life)
    DC-EOL-004: vs Cat 10 (processing intermediates != final disposal)
    DC-EOL-005: vs Scope 1 (on-site treatment != customer disposal)
    DC-EOL-006: vs Cat 13 (downstream leased assets boundary)
    DC-EOL-007: Avoided emissions from recycling reported SEPARATELY
    DC-EOL-008: Energy recovery credits reported SEPARATELY

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
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

ENGINE_ID: str = "eol_compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# SBTi minimum target coverage percentage
SBTI_MIN_TARGET_COVERAGE_PCT: Decimal = Decimal("67")

# Materiality threshold (1% of Scope 3)
MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("1")

# EU Waste Framework Directive diversion targets by year
EU_DIVERSION_TARGETS: Dict[int, Decimal] = {
    2025: Decimal("55"),
    2030: Decimal("60"),
    2035: Decimal("65"),
}

# Minimum recycling rate for ESRS E5 circular economy
ESRS_E5_MIN_RECYCLING_RATE_PCT: Decimal = Decimal("50")


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
    """Compliance check status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class ComplianceSeverity(str, Enum):
    """Severity level for compliance findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories that could overlap with Category 12."""

    CATEGORY_1 = "CATEGORY_1"    # Purchased goods (cradle-to-gate vs cradle-to-grave)
    CATEGORY_5 = "CATEGORY_5"    # Waste from own operations
    CATEGORY_10 = "CATEGORY_10"  # Processing of sold products
    CATEGORY_11 = "CATEGORY_11"  # Use of sold products
    CATEGORY_13 = "CATEGORY_13"  # Downstream leased assets
    SCOPE_1 = "SCOPE_1"          # On-site waste treatment


class BoundaryClassification(str, Enum):
    """Category boundary classification for a product EOL stream."""

    CATEGORY_12 = "CATEGORY_12"  # End-of-life of sold products (correct)
    CATEGORY_5 = "CATEGORY_5"    # Waste from own operations
    CATEGORY_10 = "CATEGORY_10"  # Processing of sold products
    CATEGORY_11 = "CATEGORY_11"  # Use-phase of sold products
    CATEGORY_13 = "CATEGORY_13"  # Downstream leased assets
    SCOPE_1 = "SCOPE_1"          # On-site treatment
    EXCLUDED = "EXCLUDED"        # Not in scope


class TreatmentPathway(str, Enum):
    """End-of-life treatment pathways."""

    LANDFILL = "landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    OPEN_BURNING = "open_burning"
    WASTEWATER = "wastewater"
    WASTE_TO_ENERGY = "waste_to_energy"


class WasteHierarchyLevel(str, Enum):
    """EU Waste Framework Directive hierarchy levels."""

    PREVENTION = "PREVENTION"
    REUSE = "REUSE"
    RECYCLING = "RECYCLING"
    RECOVERY = "RECOVERY"
    DISPOSAL = "DISPOSAL"


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
class ComplianceCheckResult:
    """Result of compliance check for one framework."""

    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


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

FRAMEWORK_WEIGHTS: Dict[ComplianceFramework, Decimal] = {
    ComplianceFramework.GHG_PROTOCOL: Decimal("1.00"),
    ComplianceFramework.ISO_14064: Decimal("0.85"),
    ComplianceFramework.CSRD_ESRS: Decimal("0.90"),
    ComplianceFramework.CDP: Decimal("0.85"),
    ComplianceFramework.SBTI: Decimal("0.80"),
    ComplianceFramework.SB_253: Decimal("0.75"),
    ComplianceFramework.GRI: Decimal("0.70"),
}

# Default enabled frameworks
DEFAULT_ENABLED_FRAMEWORKS: List[str] = [
    "GHG_PROTOCOL_SCOPE3",
    "ISO_14064",
    "CSRD_ESRS_E1",
    "CDP",
    "SBTI",
    "SB_253",
    "GRI",
]


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for End-of-Life Treatment of Sold Products (Category 12).

    Validates calculation results against 7 regulatory frameworks with
    Category 12-specific rules including product boundary enforcement,
    double-counting prevention, avoided emissions separate reporting,
    circularity metrics, and treatment method disclosure.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _enabled_frameworks: List of enabled compliance framework names
        _check_count: Running count of compliance checks performed
        _strict_mode: If True, warnings become failures

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
        self._enabled_frameworks: List[str] = list(DEFAULT_ENABLED_FRAMEWORKS)
        self._check_count: int = 0
        self._strict_mode: bool = False
        self._materiality_threshold: Decimal = MATERIALITY_THRESHOLD_PCT

        # Attempt to load from config module
        try:
            from greenlang.end_of_life_treatment.config import get_config
            config = get_config()
            if config and hasattr(config, "compliance"):
                comp_cfg = config.compliance
                if hasattr(comp_cfg, "get_frameworks"):
                    self._enabled_frameworks = comp_cfg.get_frameworks()
                if hasattr(comp_cfg, "strict_mode"):
                    self._strict_mode = comp_cfg.strict_mode
                if hasattr(comp_cfg, "materiality_threshold"):
                    self._materiality_threshold = Decimal(
                        str(comp_cfg.materiality_threshold)
                    ) * Decimal("100")
        except (ImportError, AttributeError, Exception) as exc:
            logger.debug(
                "Config not available, using defaults: %s", exc
            )

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
                treatment_breakdown, material_breakdown, method, ef_sources,
                avoided_emissions, biogenic_co2, etc.

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

    def check_compliance(
        self,
        result: dict,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        High-level compliance check entry point.

        Runs all framework checks plus double-counting prevention,
        then returns a comprehensive report.

        Args:
            result: Calculation result dictionary.
            frameworks: Optional list of framework names to check.
                If None, checks all enabled frameworks.

        Returns:
            Comprehensive compliance report dictionary.
        """
        if frameworks is not None:
            original = self._enabled_frameworks
            self._enabled_frameworks = frameworks

        try:
            framework_results = self.check_all_frameworks(result)
            dc_findings = self.check_double_counting(result)
            completeness = self.validate_completeness(result)
            avoided = self.validate_avoided_emissions_reporting(result)
            summary = self.get_compliance_summary(framework_results)

            summary["double_counting_findings"] = dc_findings
            summary["completeness"] = completeness
            summary["avoided_emissions_reporting"] = avoided
            return summary
        finally:
            if frameworks is not None:
                self._enabled_frameworks = original

    # ==========================================================================
    # Framework 1: GHG Protocol Scope 3 (8 rules)
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 12).

        Rules:
            GHG-EOL-001: Total CO2e present and positive
            GHG-EOL-002: Treatment method breakdown provided
            GHG-EOL-003: Material composition documented
            GHG-EOL-004: Avoided emissions reported separately
            GHG-EOL-005: Biogenic CO2 separated from fossil CO2
            GHG-EOL-006: Calculation method justified
            GHG-EOL-007: Data quality indicator present
            GHG-EOL-008: Product boundary (sold products only)

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

        # GHG-EOL-001: Total CO2e present and positive
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GHG-EOL-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-EOL-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Ensure total_co2e is calculated and > 0.",
                regulation_reference="GHG Protocol Scope 3, Ch 12",
            )

        # GHG-EOL-002: Treatment method breakdown
        treatment_breakdown = (
            result.get("treatment_breakdown")
            or result.get("by_treatment")
            or result.get("treatment_distribution")
        )
        if treatment_breakdown and isinstance(treatment_breakdown, dict) and len(treatment_breakdown) > 0:
            state.add_pass(
                "GHG-EOL-002",
                "Treatment method breakdown provided",
            )
        else:
            state.add_fail(
                "GHG-EOL-002",
                "Treatment method breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by end-of-life treatment method "
                    "(landfill, incineration, recycling, composting, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 12.1",
            )

        # GHG-EOL-003: Material composition documented
        material_breakdown = (
            result.get("material_breakdown")
            or result.get("by_material")
            or result.get("material_composition")
        )
        if material_breakdown and isinstance(material_breakdown, dict) and len(material_breakdown) > 0:
            state.add_pass(
                "GHG-EOL-003",
                "Material composition documented",
            )
        else:
            state.add_fail(
                "GHG-EOL-003",
                "Material composition not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the material composition of sold products "
                    "(e.g., plastics, metals, paper, glass, organic, electronics)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 12",
            )

        # GHG-EOL-004: Avoided emissions reported separately
        avoided_emissions = result.get("avoided_emissions") or result.get("avoided_co2e")
        avoided_reported_separately = result.get("avoided_reported_separately", False)

        if avoided_emissions is not None:
            if avoided_reported_separately:
                state.add_pass(
                    "GHG-EOL-004",
                    "Avoided emissions reported separately (not netted off)",
                )
            else:
                state.add_fail(
                    "GHG-EOL-004",
                    "Avoided emissions not confirmed as separately reported",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Avoided emissions from recycling and energy recovery MUST be "
                        "reported separately from gross emissions, not netted off. "
                        "Set avoided_reported_separately=True to confirm."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 12, Note",
                )
        else:
            state.add_pass(
                "GHG-EOL-004",
                "No avoided emissions reported (no netting risk)",
            )

        # GHG-EOL-005: Biogenic CO2 separated
        co2_biogenic = result.get("co2_biogenic_kg") or result.get("biogenic_co2")
        co2_fossil = result.get("co2_fossil_kg") or result.get("fossil_co2")
        if co2_biogenic is not None or co2_fossil is not None:
            state.add_pass(
                "GHG-EOL-005",
                "Biogenic and/or fossil CO2 separated",
            )
        else:
            state.add_warning(
                "GHG-EOL-005",
                "Biogenic vs fossil CO2 not separated",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Separate biogenic CO2 (from organic/biomass incineration) from "
                    "fossil CO2 (from plastic/synthetic incineration). Biogenic CO2 "
                    "should be reported as a memo item, not in total CO2e."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 12",
            )

        # GHG-EOL-006: Calculation method justified
        method = (
            result.get("method")
            or result.get("calculation_method")
            or result.get("methodology")
        )
        if method:
            state.add_pass("GHG-EOL-006", "Calculation method documented")
        else:
            state.add_fail(
                "GHG-EOL-006",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used: waste-type-specific, "
                    "average-data, or producer-specific (EPD-based)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 12.1",
            )

        # GHG-EOL-007: Data quality indicator
        dqi_score = (
            result.get("dqi_score")
            or result.get("data_quality_score")
            or result.get("data_quality")
        )
        if dqi_score is not None:
            state.add_pass("GHG-EOL-007", "Data quality indicator present")
        else:
            state.add_warning(
                "GHG-EOL-007",
                "Data quality indicator (DQI) not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across quality dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-EOL-008: Product boundary (sold products only)
        product_boundary = result.get("product_boundary") or result.get("boundary")
        includes_own_operations = result.get("includes_own_operations_waste", False)
        if includes_own_operations:
            state.add_fail(
                "GHG-EOL-008",
                "Product boundary includes own operations waste",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Category 12 covers ONLY products sold by the reporting company. "
                    "Own operations waste belongs in Category 5. Remove any waste "
                    "from company operations."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 12 vs Ch 5",
            )
        elif product_boundary and str(product_boundary).lower() == "sold_products":
            state.add_pass(
                "GHG-EOL-008",
                "Product boundary correctly set to sold products",
            )
        else:
            state.add_warning(
                "GHG-EOL-008",
                "Product boundary not explicitly documented as 'sold_products'",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Explicitly document that the boundary covers only "
                    "products sold by the reporting company in the reporting year."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 12",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 2: ISO 14064 (6 rules)
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018.

        Rules:
            ISO-EOL-001: Total CO2e present
            ISO-EOL-002: Uncertainty analysis
            ISO-EOL-003: Base year documented
            ISO-EOL-004: Methodology described
            ISO-EOL-005: Reporting period defined
            ISO-EOL-006: Verification status

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

        # ISO-EOL-001: Total emissions present
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("ISO-EOL-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-EOL-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total CO2e emissions.",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-EOL-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-EOL-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-EOL-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis. End-of-life "
                    "treatment assumptions carry significant uncertainty "
                    "(treatment pathway splits, regional variation)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-EOL-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-EOL-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-EOL-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison "
                    "and trend analysis."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-EOL-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-EOL-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-EOL-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "treatment pathway assumptions, emission factors, and "
                    "data sources."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-EOL-005: Reporting period defined
        reporting_period = (
            result.get("reporting_period")
            or result.get("period")
            or result.get("reporting_year")
        )
        if reporting_period:
            state.add_pass("ISO-EOL-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-EOL-005",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation="Specify the reporting period (e.g., 2025, FY2025).",
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-EOL-006: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("ISO-EOL-006", "Verification status documented")
        else:
            state.add_warning(
                "ISO-EOL-006",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 12 emissions have been "
                    "verified by an independent third party."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 10",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 3: CSRD/ESRS E1 + E5 (8 rules)
    # ==========================================================================

    def check_csrd_esrs(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CSRD ESRS E1 Climate Change + E5 Circular Economy.

        Rules:
            CSRD-EOL-001: Total CO2e by category
            CSRD-EOL-002: Methodology description
            CSRD-EOL-003: Targets documented
            CSRD-EOL-004: Product breakdown present
            CSRD-EOL-005: ESRS E5 recycling rate
            CSRD-EOL-006: ESRS E5 diversion rate
            CSRD-EOL-007: ESRS E5 waste hierarchy compliance
            CSRD-EOL-008: ESRS E5 resource outflows documented

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

        # CSRD-EOL-001: Total emissions by category
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CSRD-EOL-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-EOL-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Scope 3 Category 12 emissions.",
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-EOL-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-EOL-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-EOL-002",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for end-of-life "
                    "treatment emissions as required by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-EOL-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-EOL-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-EOL-003",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for end-of-life treatment "
                    "(e.g., increase recyclability, reduce landfill share)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-EOL-004: Product breakdown present
        product_breakdown = (
            result.get("product_breakdown")
            or result.get("by_product")
            or result.get("by_category")
        )
        if product_breakdown and isinstance(product_breakdown, dict) and len(product_breakdown) > 0:
            state.add_pass("CSRD-EOL-004", "Product breakdown present")
        else:
            state.add_fail(
                "CSRD-EOL-004",
                "Product breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by product category "
                    "to meet ESRS E1 disclosure requirements."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-EOL-005: ESRS E5 recycling rate
        recycling_rate = result.get("recycling_rate") or result.get("recycling_rate_pct")
        if recycling_rate is not None:
            rate = Decimal(str(recycling_rate))
            if rate >= ESRS_E5_MIN_RECYCLING_RATE_PCT:
                state.add_pass(
                    "CSRD-EOL-005",
                    f"Recycling rate ({rate}%) meets ESRS E5 threshold",
                )
            else:
                state.add_warning(
                    "CSRD-EOL-005",
                    f"Recycling rate ({rate}%) below {ESRS_E5_MIN_RECYCLING_RATE_PCT}% target",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Improve product recyclability to increase the recycling "
                        "rate. ESRS E5 requires disclosure of circular economy metrics."
                    ),
                    regulation_reference="ESRS E5-5, para 37",
                )
        else:
            state.add_warning(
                "CSRD-EOL-005",
                "Recycling rate not reported",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report the recycling rate for sold products "
                    "as required by ESRS E5 circular economy metrics."
                ),
                regulation_reference="ESRS E5-5, para 37",
            )

        # CSRD-EOL-006: ESRS E5 diversion rate
        diversion_rate = result.get("diversion_rate") or result.get("diversion_rate_pct")
        if diversion_rate is not None:
            rate = Decimal(str(diversion_rate))
            # Check against applicable EU target
            reporting_year = result.get("reporting_year", 2025)
            target = self._get_eu_diversion_target(reporting_year)
            if rate >= target:
                state.add_pass(
                    "CSRD-EOL-006",
                    f"Diversion rate ({rate}%) meets EU target ({target}%)",
                )
            else:
                state.add_warning(
                    "CSRD-EOL-006",
                    f"Diversion rate ({rate}%) below EU target ({target}%)",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        f"Increase landfill diversion to meet the EU Waste "
                        f"Framework Directive target of {target}%."
                    ),
                    regulation_reference="ESRS E5-5; EU WFD 2008/98/EC",
                )
        else:
            state.add_warning(
                "CSRD-EOL-006",
                "Diversion rate not reported",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report the landfill diversion rate "
                    "(% of waste diverted from landfill)."
                ),
                regulation_reference="ESRS E5-5, para 37",
            )

        # CSRD-EOL-007: Waste hierarchy compliance
        waste_hierarchy = result.get("waste_hierarchy") or result.get("hierarchy_compliance")
        if waste_hierarchy:
            state.add_pass("CSRD-EOL-007", "Waste hierarchy compliance documented")
        else:
            state.add_warning(
                "CSRD-EOL-007",
                "Waste hierarchy compliance not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document alignment with the EU waste hierarchy "
                    "(prevention > reuse > recycling > recovery > disposal)."
                ),
                regulation_reference="ESRS E5; EU WFD Art 4",
            )

        # CSRD-EOL-008: Resource outflows documented
        resource_outflows = result.get("resource_outflows") or result.get("material_outflows")
        if resource_outflows:
            state.add_pass("CSRD-EOL-008", "Resource outflows documented")
        else:
            state.add_warning(
                "CSRD-EOL-008",
                "Resource outflows not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document resource outflows (total waste generated by products "
                    "sold, by treatment type) as required by ESRS E5."
                ),
                regulation_reference="ESRS E5-5, para 38",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 4: CDP Climate Change (5 rules)
    # ==========================================================================

    def check_cdp(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change Questionnaire (C6.5 Category 12).

        Rules:
            CDP-EOL-001: Total CO2e present for C6.5
            CDP-EOL-002: Product breakdown
            CDP-EOL-003: Data quality documented
            CDP-EOL-004: Methodology documented
            CDP-EOL-005: Verification status

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

        # CDP-EOL-001: Total emissions present
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CDP-EOL-001", "Total CO2e reported for C6.5")
        else:
            state.add_fail(
                "CDP-EOL-001",
                "Total CO2e missing or zero for CDP C6.5 Category 12",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 12 emissions to CDP C6.5.",
                regulation_reference="CDP CC Module 7, C6.5",
            )

        # CDP-EOL-002: Product breakdown
        product_breakdown = (
            result.get("product_breakdown")
            or result.get("by_product")
            or result.get("by_category")
        )
        if product_breakdown and isinstance(product_breakdown, dict) and len(product_breakdown) > 0:
            state.add_pass("CDP-EOL-002", "Product breakdown present for CDP")
        else:
            state.add_fail(
                "CDP-EOL-002",
                "Product breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by product category "
                    "for CDP C6.5 reporting."
                ),
                regulation_reference="CDP CC Module 7, C6.5",
            )

        # CDP-EOL-003: Data quality documented
        dqi_score = (
            result.get("dqi_score")
            or result.get("data_quality_score")
            or result.get("data_quality")
        )
        if dqi_score is not None:
            state.add_pass("CDP-EOL-003", "Data quality documented")
        else:
            state.add_warning(
                "CDP-EOL-003",
                "Data quality not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document data quality indicators for Category 12 emissions. "
                    "CDP scores data quality in C6.5."
                ),
                regulation_reference="CDP CC Module 7, C6.5",
            )

        # CDP-EOL-004: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CDP-EOL-004", "Methodology documented for CDP")
        else:
            state.add_fail(
                "CDP-EOL-004",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology (waste-type-specific, "
                    "average-data, or producer-specific) for CDP C6.5."
                ),
                regulation_reference="CDP CC Module 7, C6.5",
            )

        # CDP-EOL-005: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-EOL-005", "Verification status documented")
        else:
            state.add_warning(
                "CDP-EOL-005",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 12 emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module 10, Q10.1",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 5: SBTi (6 rules)
    # ==========================================================================

    def check_sbti(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative.

        Rules:
            SBTI-EOL-001: Total CO2e present
            SBTI-EOL-002: Target coverage (67% min of Scope 3)
            SBTI-EOL-003: Base year documented
            SBTI-EOL-004: Progress tracking
            SBTI-EOL-005: Materiality assessment
            SBTI-EOL-006: Reduction pathway documented

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

        # SBTI-EOL-001: Total emissions present
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SBTI-EOL-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-EOL-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate total Category 12 emissions for SBTi target boundary.",
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-EOL-002: Target coverage (67% min)
        target_coverage = (
            result.get("target_coverage")
            or result.get("sbti_coverage")
        )
        if target_coverage is not None:
            try:
                coverage_pct = Decimal(str(target_coverage))
                if coverage_pct >= SBTI_MIN_TARGET_COVERAGE_PCT:
                    state.add_pass(
                        "SBTI-EOL-002",
                        f"Target coverage ({coverage_pct}%) meets 67% minimum",
                    )
                else:
                    state.add_fail(
                        "SBTI-EOL-002",
                        f"Target coverage ({coverage_pct}%) below 67% minimum",
                        ComplianceSeverity.HIGH,
                        recommendation=(
                            "SBTi requires minimum 67% of Scope 3 emissions to be "
                            "covered by targets. Increase category coverage."
                        ),
                        regulation_reference="SBTi Criteria v5.1, C20",
                    )
            except (InvalidOperation, ValueError):
                state.add_warning(
                    "SBTI-EOL-002",
                    "Could not parse target coverage value",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-EOL-002",
                "SBTi target coverage not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document what percentage of Scope 3 emissions is covered "
                    "by SBTi targets (minimum 67% required)."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-EOL-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("SBTI-EOL-003", "Base year documented for SBTi")
        else:
            state.add_fail(
                "SBTI-EOL-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for SBTi target setting and "
                    "progress tracking."
                ),
                regulation_reference="SBTi Criteria v5.1, C7",
            )

        # SBTI-EOL-004: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-EOL-004", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-EOL-004",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 12 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-EOL-005: Materiality assessment
        total_scope3 = result.get("total_scope3_co2e") or result.get("total_scope3")
        if total_co2e and total_scope3:
            try:
                cat12_val = Decimal(str(total_co2e))
                scope3_val = Decimal(str(total_scope3))
                if scope3_val > 0:
                    cat12_pct = (cat12_val / scope3_val * Decimal("100")).quantize(
                        _QUANT_2DP, rounding=ROUNDING
                    )
                    threshold = self._materiality_threshold
                    if cat12_pct >= threshold:
                        state.add_pass(
                            "SBTI-EOL-005",
                            f"Category 12 is material ({cat12_pct}% of Scope 3)",
                        )
                    else:
                        state.add_warning(
                            "SBTI-EOL-005",
                            f"Category 12 below materiality threshold ({cat12_pct}% of Scope 3)",
                            ComplianceSeverity.LOW,
                            recommendation=(
                                "Category 12 may not need to be included in SBTi "
                                "target boundary if below materiality threshold."
                            ),
                        )
                else:
                    state.add_warning(
                        "SBTI-EOL-005",
                        "Scope 3 total is zero; cannot assess materiality",
                        ComplianceSeverity.LOW,
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-EOL-005",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-EOL-005",
                "Total Scope 3 emissions not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 12 materiality."
                ),
            )

        # SBTI-EOL-006: Reduction pathway documented
        reduction_pathway = (
            result.get("reduction_pathway")
            or result.get("reduction_actions")
            or result.get("actions")
        )
        if reduction_pathway:
            state.add_pass("SBTI-EOL-006", "Reduction pathway documented")
        else:
            state.add_warning(
                "SBTI-EOL-006",
                "Reduction pathway not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document a reduction pathway for Category 12 emissions "
                    "(e.g., design for recyclability, eliminate problematic "
                    "materials, extended producer responsibility)."
                ),
                regulation_reference="SBTi Criteria v5.1",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 6: SB 253 (5 rules)
    # ==========================================================================

    def check_sb_253(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Rules:
            SB253-EOL-001: Total CO2e present
            SB253-EOL-002: Methodology documented
            SB253-EOL-003: Assurance opinion
            SB253-EOL-004: Threshold (1% materiality)
            SB253-EOL-005: Materiality classification

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

        # SB253-EOL-001: Total emissions present
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SB253-EOL-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-EOL-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 12 emissions for SB 253 compliance.",
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-EOL-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-EOL-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-EOL-002",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-EOL-003: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-EOL-003", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-EOL-003",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-EOL-004: Threshold (1% materiality)
        total_scope3 = result.get("total_scope3_co2e") or result.get("total_scope3")
        if total_co2e and total_scope3:
            try:
                cat12_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                ).quantize(_QUANT_2DP, rounding=ROUNDING)
                if cat12_pct > MATERIALITY_THRESHOLD_PCT:
                    state.add_pass(
                        "SB253-EOL-004",
                        f"Category 12 exceeds 1% materiality ({cat12_pct}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SB253-EOL-004",
                        f"Category 12 below 1% materiality threshold ({cat12_pct}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 12 is below 1% of total Scope 3. "
                            "Consider whether separate reporting is warranted."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SB253-EOL-004",
                    "Could not calculate materiality",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SB253-EOL-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 12 materiality against 1% threshold."
                ),
            )

        # SB253-EOL-005: Materiality classification
        materiality = result.get("materiality_classification") or result.get("materiality")
        if materiality:
            state.add_pass("SB253-EOL-005", "Materiality classification documented")
        else:
            state.add_warning(
                "SB253-EOL-005",
                "Materiality classification not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Classify Category 12 as material or immaterial and "
                    "document the rationale."
                ),
                regulation_reference="SB 253, Section 38532(c)",
            )

        return state.to_result()

    # ==========================================================================
    # Framework 7: GRI 305 + 306 (4 rules)
    # ==========================================================================

    def check_gri(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305 Emissions + GRI 306 Waste Standards.

        Rules:
            GRI-EOL-001: Total CO2e in metric tonnes
            GRI-EOL-002: Gases included documented
            GRI-EOL-003: Base year present
            GRI-EOL-004: Emission factor sources documented

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

        # GRI-EOL-001: Total emissions present
        total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GRI-EOL-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-EOL-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 12 emissions in metric tonnes CO2e.",
                regulation_reference="GRI 305-3",
            )

        # GRI-EOL-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-EOL-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-EOL-002",
                "Gases included not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, or CO2e aggregate). End-of-life treatment "
                    "commonly involves CO2 (fossil), CH4 (landfill/AD), and N2O "
                    "(composting/wastewater)."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-EOL-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-EOL-003", "Base year present")
        else:
            state.add_warning(
                "GRI-EOL-003",
                "Base year not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-EOL-004: Source of emission factors
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-EOL-004", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-EOL-004",
                "Emission factor sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., EPA WARM v16, DEFRA 2024, IPCC 2006/2019)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention (8 rules)
    # ==========================================================================

    def check_double_counting(self, result: dict) -> List[Dict[str, Any]]:
        """
        Validate 8 double-counting prevention rules for Category 12.

        Rules:
            DC-EOL-001: vs Cat 5 (own waste != customer disposal)
            DC-EOL-002: vs Cat 1 (upstream cradle-to-gate excludes post-consumer EOL)
            DC-EOL-003: vs Cat 11 (use-phase != end-of-life)
            DC-EOL-004: vs Cat 10 (processing intermediates != final disposal)
            DC-EOL-005: vs Scope 1 (on-site treatment != customer disposal)
            DC-EOL-006: vs Cat 13 (downstream leased assets boundary)
            DC-EOL-007: Avoided emissions from recycling reported SEPARATELY
            DC-EOL-008: Energy recovery credits reported SEPARATELY

        Args:
            result: Calculation result dictionary containing fields like
                includes_own_operations_waste, cradle_to_grave_in_cat1,
                includes_use_phase, includes_processing_waste,
                includes_onsite_treatment, includes_leased_assets,
                avoided_reported_separately, energy_credits_reported_separately,
                products list, etc.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and recommendation.

        Example:
            >>> findings = engine.check_double_counting(calc_result)
            >>> len(findings)
            0  # No double-counting issues
        """
        findings: List[Dict[str, Any]] = []

        # DC-EOL-001: vs Category 5 (own waste != customer disposal)
        includes_own_waste = result.get("includes_own_operations_waste", False)
        if includes_own_waste:
            findings.append({
                "rule_code": "DC-EOL-001",
                "description": (
                    "Result includes own operations waste. Category 12 covers "
                    "ONLY end-of-life treatment of products SOLD to customers. "
                    "Own operations waste belongs in Category 5."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "category": DoubleCountingCategory.CATEGORY_5.value,
                "recommendation": (
                    "Remove own operations waste from Category 12. "
                    "Report it under Category 5 (Waste Generated in Operations)."
                ),
            })

        # DC-EOL-002: vs Category 1 (cradle-to-gate excludes post-consumer EOL)
        cradle_to_grave_in_cat1 = result.get("cradle_to_grave_in_cat1", False)
        if cradle_to_grave_in_cat1:
            findings.append({
                "rule_code": "DC-EOL-002",
                "description": (
                    "Category 1 includes cradle-to-grave emissions. If Cat 1 "
                    "already includes post-consumer end-of-life, those emissions "
                    "must NOT also be counted in Category 12."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "category": DoubleCountingCategory.CATEGORY_1.value,
                "recommendation": (
                    "If Category 1 (Purchased Goods) uses cradle-to-grave factors, "
                    "exclude the end-of-life portion from Category 12 to avoid "
                    "double-counting. Use cradle-to-gate factors in Cat 1 instead."
                ),
            })

        # DC-EOL-003: vs Category 11 (use-phase != end-of-life)
        includes_use_phase = result.get("includes_use_phase", False)
        if includes_use_phase:
            findings.append({
                "rule_code": "DC-EOL-003",
                "description": (
                    "Result includes use-phase emissions. Category 12 covers "
                    "ONLY the end-of-life treatment, not the use phase. "
                    "Use-phase belongs in Category 11."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "category": DoubleCountingCategory.CATEGORY_11.value,
                "recommendation": (
                    "Remove use-phase emissions from Category 12. "
                    "Report them under Category 11 (Use of Sold Products)."
                ),
            })

        # DC-EOL-004: vs Category 10 (processing intermediates != final disposal)
        includes_processing = result.get("includes_processing_waste", False)
        if includes_processing:
            findings.append({
                "rule_code": "DC-EOL-004",
                "description": (
                    "Result includes intermediate processing waste. Category 12 "
                    "covers ONLY final consumer disposal, not intermediate "
                    "processing. Processing waste belongs in Category 10."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "category": DoubleCountingCategory.CATEGORY_10.value,
                "recommendation": (
                    "Remove intermediate processing waste from Category 12. "
                    "Report it under Category 10 (Processing of Sold Products)."
                ),
            })

        # DC-EOL-005: vs Scope 1 (on-site treatment != customer disposal)
        includes_onsite = result.get("includes_onsite_treatment", False)
        if includes_onsite:
            findings.append({
                "rule_code": "DC-EOL-005",
                "description": (
                    "Result includes on-site waste treatment. On-site treatment "
                    "belongs in Scope 1/2, not Category 12. Category 12 covers "
                    "only third-party treatment of products after sale."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "Remove on-site waste treatment from Category 12. "
                    "Report under Scope 1 (direct emissions) if applicable."
                ),
            })

        # DC-EOL-006: vs Category 13 (downstream leased assets boundary)
        includes_leased = result.get("includes_leased_assets", False)
        if includes_leased:
            findings.append({
                "rule_code": "DC-EOL-006",
                "description": (
                    "Result includes downstream leased assets. The end-of-life "
                    "of leased assets should be reported under Category 13 "
                    "(Downstream Leased Assets), not Category 12."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "category": DoubleCountingCategory.CATEGORY_13.value,
                "recommendation": (
                    "Remove leased asset end-of-life from Category 12. "
                    "Report under Category 13 (Downstream Leased Assets)."
                ),
            })

        # DC-EOL-007: Avoided emissions from recycling reported SEPARATELY
        avoided_emissions = result.get("avoided_emissions") or result.get("avoided_co2e")
        avoided_separate = result.get("avoided_reported_separately", False)
        if avoided_emissions is not None:
            try:
                avoided_val = Decimal(str(avoided_emissions))
                if avoided_val != Decimal("0") and not avoided_separate:
                    findings.append({
                        "rule_code": "DC-EOL-007",
                        "description": (
                            "Avoided emissions from recycling are present but NOT "
                            "confirmed as separately reported. GHG Protocol requires "
                            "avoided emissions to be reported SEPARATELY from gross "
                            "emissions, never netted off."
                        ),
                        "severity": ComplianceSeverity.CRITICAL.value,
                        "category": "AVOIDED_EMISSIONS",
                        "recommendation": (
                            "Report avoided emissions from recycling as a separate "
                            "line item. Do NOT subtract them from gross Category 12 "
                            "emissions. Set avoided_reported_separately=True."
                        ),
                    })
            except (InvalidOperation, ValueError):
                pass

        # DC-EOL-008: Energy recovery credits reported SEPARATELY
        energy_credits = result.get("energy_recovery_credits") or result.get("wte_credits")
        energy_separate = result.get("energy_credits_reported_separately", False)
        if energy_credits is not None:
            try:
                credits_val = Decimal(str(energy_credits))
                if credits_val != Decimal("0") and not energy_separate:
                    findings.append({
                        "rule_code": "DC-EOL-008",
                        "description": (
                            "Energy recovery credits are present but NOT confirmed "
                            "as separately reported. GHG Protocol requires energy "
                            "recovery credits to be reported SEPARATELY, never "
                            "netted off gross emissions."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "category": "ENERGY_RECOVERY",
                        "recommendation": (
                            "Report energy recovery credits (from waste-to-energy) "
                            "as a separate line item. Do NOT subtract from gross "
                            "Category 12 emissions."
                        ),
                    })
            except (InvalidOperation, ValueError):
                pass

        logger.info(
            "Double-counting check: %d findings",
            len(findings),
        )

        return findings

    # ==========================================================================
    # Boundary Validation
    # ==========================================================================

    def validate_boundary(self, product: dict) -> Dict[str, Any]:
        """
        Determine if a product EOL stream belongs in Category 12 vs other categories.

        Classification Rules:
            - Sold product EOL -> Category 12
            - Own operations waste -> Category 5
            - Use-phase emissions -> Category 11
            - Processing waste -> Category 10
            - Leased asset EOL -> Category 13
            - On-site treatment -> Scope 1

        Args:
            product: Product dictionary with fields: product_type, sold_to_customer,
                is_leased, is_intermediate, waste_from_operations, treated_onsite.

        Returns:
            Dictionary with classification, reason, and warnings.

        Example:
            >>> result = engine.validate_boundary({
            ...     "product_type": "consumer_electronics",
            ...     "sold_to_customer": True,
            ... })
            >>> result["classification"]
            'CATEGORY_12'
        """
        warnings: List[str] = []

        # Rule 1: Own operations waste -> Category 5
        if product.get("waste_from_operations", False):
            return {
                "classification": BoundaryClassification.CATEGORY_5.value,
                "reason": (
                    "Waste from own operations belongs in Category 5 "
                    "(Waste Generated in Operations), not Category 12."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 2: On-site treatment -> Scope 1
        if product.get("treated_onsite", False):
            return {
                "classification": BoundaryClassification.SCOPE_1.value,
                "reason": (
                    "On-site waste treatment belongs in Scope 1 "
                    "(direct emissions), not Category 12."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 3: Leased asset -> Category 13
        if product.get("is_leased", False):
            return {
                "classification": BoundaryClassification.CATEGORY_13.value,
                "reason": (
                    "End-of-life of leased assets belongs in Category 13 "
                    "(Downstream Leased Assets), not Category 12."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 4: Intermediate product -> Category 10
        if product.get("is_intermediate", False):
            return {
                "classification": BoundaryClassification.CATEGORY_10.value,
                "reason": (
                    "Processing waste from intermediate products belongs in "
                    "Category 10 (Processing of Sold Products), not Category 12."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Rule 5: Sold to customer -> Category 12
        sold_to_customer = product.get("sold_to_customer", True)
        if sold_to_customer:
            # Additional warnings for ambiguous cases
            if not product.get("product_type"):
                warnings.append(
                    "Product type not specified. Consider documenting "
                    "product type for audit trail."
                )

            if product.get("has_take_back_program", False):
                warnings.append(
                    "Product has a take-back program. Verify that take-back "
                    "volumes are excluded from Category 12 if treated on-site."
                )

            return {
                "classification": BoundaryClassification.CATEGORY_12.value,
                "reason": (
                    "End-of-life treatment of sold products belongs in "
                    "Category 12."
                ),
                "warnings": warnings,
                "product": product,
            }

        # Default: Excluded
        return {
            "classification": BoundaryClassification.EXCLUDED.value,
            "reason": (
                "Product could not be classified into a Scope 3 category. "
                "Verify the product is sold to external customers."
            ),
            "warnings": warnings,
            "product": product,
        }

    # ==========================================================================
    # Completeness Validation
    # ==========================================================================

    def validate_completeness(self, result: dict) -> Dict[str, Any]:
        """
        Assess the completeness of Category 12 reporting.

        Scores completeness across multiple dimensions:
        - Product coverage (% of sold product categories covered)
        - Material coverage (% of material types identified)
        - Treatment coverage (% of treatment methods addressed)
        - Regional coverage (% of sales regions covered)
        - Temporal coverage (full reporting period)

        Args:
            result: Calculation result dictionary.

        Returns:
            Completeness assessment dictionary with dimension scores
            and overall completeness percentage.
        """
        dimensions: Dict[str, Dict[str, Any]] = {}

        # Product coverage
        products_covered = result.get("products_covered", 0)
        products_total = result.get("products_total", 0)
        if products_total > 0:
            pct = (Decimal(str(products_covered)) / Decimal(str(products_total)) * Decimal("100")).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            dimensions["product_coverage"] = {
                "score": float(pct),
                "covered": products_covered,
                "total": products_total,
                "status": "complete" if pct >= Decimal("95") else "incomplete",
            }
        else:
            dimensions["product_coverage"] = {
                "score": 0.0,
                "covered": 0,
                "total": 0,
                "status": "unknown",
            }

        # Material coverage
        material_breakdown = result.get("material_breakdown") or result.get("by_material")
        if material_breakdown and isinstance(material_breakdown, dict):
            material_count = len(material_breakdown)
            dimensions["material_coverage"] = {
                "score": 100.0 if material_count >= 3 else material_count / 3 * 100,
                "materials_identified": material_count,
                "status": "complete" if material_count >= 3 else "partial",
            }
        else:
            dimensions["material_coverage"] = {
                "score": 0.0,
                "materials_identified": 0,
                "status": "missing",
            }

        # Treatment coverage
        treatment_breakdown = result.get("treatment_breakdown") or result.get("by_treatment")
        if treatment_breakdown and isinstance(treatment_breakdown, dict):
            treatment_count = len(treatment_breakdown)
            dimensions["treatment_coverage"] = {
                "score": 100.0 if treatment_count >= 2 else treatment_count / 2 * 100,
                "treatments_identified": treatment_count,
                "status": "complete" if treatment_count >= 2 else "partial",
            }
        else:
            dimensions["treatment_coverage"] = {
                "score": 0.0,
                "treatments_identified": 0,
                "status": "missing",
            }

        # Regional coverage
        regions_covered = result.get("regions_covered", [])
        regions_total = result.get("regions_total", [])
        if regions_total and len(regions_total) > 0:
            region_pct = len(regions_covered) / len(regions_total) * 100
            dimensions["regional_coverage"] = {
                "score": region_pct,
                "regions_covered": len(regions_covered),
                "regions_total": len(regions_total),
                "status": "complete" if region_pct >= 90 else "partial",
            }
        else:
            dimensions["regional_coverage"] = {
                "score": 100.0,  # If no regions specified, assume complete
                "regions_covered": 0,
                "regions_total": 0,
                "status": "not_applicable",
            }

        # Temporal coverage
        reporting_period = result.get("reporting_period") or result.get("reporting_year")
        dimensions["temporal_coverage"] = {
            "score": 100.0 if reporting_period else 0.0,
            "reporting_period": reporting_period,
            "status": "complete" if reporting_period else "missing",
        }

        # Overall completeness
        scores = [d["score"] for d in dimensions.values()]
        overall = sum(scores) / len(scores) if scores else 0.0

        return {
            "overall_completeness_pct": round(overall, 2),
            "dimensions": dimensions,
            "status": (
                "complete" if overall >= 90.0
                else "partial" if overall >= 50.0
                else "incomplete"
            ),
        }

    # ==========================================================================
    # Avoided Emissions Validation
    # ==========================================================================

    def validate_avoided_emissions_reporting(self, result: dict) -> Dict[str, Any]:
        """
        Ensure avoided emissions (recycling credits, energy recovery credits)
        are reported separately per GHG Protocol guidance.

        Args:
            result: Calculation result dictionary.

        Returns:
            Dictionary with validation status and details.
        """
        issues: List[str] = []
        status = "compliant"

        # Check recycling avoided emissions
        avoided_emissions = result.get("avoided_emissions") or result.get("avoided_co2e")
        if avoided_emissions is not None:
            try:
                val = Decimal(str(avoided_emissions))
                if val != Decimal("0"):
                    total_co2e = result.get("total_co2e") or result.get("total_co2e_kg")
                    if total_co2e is not None:
                        total = Decimal(str(total_co2e))
                        # Check if total appears to be net (negative or suspiciously low)
                        if total < Decimal("0"):
                            issues.append(
                                "Total CO2e is negative, suggesting avoided emissions "
                                "have been netted off. Report gross emissions separately."
                            )
                            status = "non_compliant"

                    if not result.get("avoided_reported_separately", False):
                        issues.append(
                            "Avoided emissions present but not confirmed as "
                            "separately reported."
                        )
                        status = "non_compliant"
            except (InvalidOperation, ValueError):
                issues.append("Could not parse avoided emissions value.")
                status = "warning"

        # Check energy recovery credits
        energy_credits = result.get("energy_recovery_credits") or result.get("wte_credits")
        if energy_credits is not None:
            try:
                val = Decimal(str(energy_credits))
                if val != Decimal("0") and not result.get("energy_credits_reported_separately", False):
                    issues.append(
                        "Energy recovery credits present but not confirmed as "
                        "separately reported."
                    )
                    status = "non_compliant"
            except (InvalidOperation, ValueError):
                issues.append("Could not parse energy recovery credits value.")
                status = "warning"

        return {
            "status": status,
            "issues": issues,
            "avoided_emissions": str(avoided_emissions) if avoided_emissions is not None else None,
            "energy_recovery_credits": str(energy_credits) if energy_credits is not None else None,
            "avoided_reported_separately": result.get("avoided_reported_separately", False),
            "energy_credits_reported_separately": result.get("energy_credits_reported_separately", False),
        }

    # ==========================================================================
    # Summary & Report Generation
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
        critical_findings: List[Dict[str, Any]] = []
        high_findings: List[Dict[str, Any]] = []
        medium_findings: List[Dict[str, Any]] = []
        low_findings: List[Dict[str, Any]] = []

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

        # Priority recommendation if score is low
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

    def generate_report(self, results: Dict[str, ComplianceCheckResult]) -> Dict[str, Any]:
        """
        Generate a comprehensive compliance report including summary,
        double-counting assessment, and recommendations.

        This is a convenience method that wraps get_compliance_summary with
        additional metadata and formatting.

        Args:
            results: Dictionary mapping framework name to ComplianceCheckResult.

        Returns:
            Full compliance report dictionary.
        """
        summary = self.get_compliance_summary(results)

        report = {
            "report_type": "eol_compliance_report",
            "agent_id": "GL-MRV-S3-012",
            "agent_component": "AGENT-MRV-025",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "category": "Scope 3 Category 12 - End-of-Life Treatment of Sold Products",
            **summary,
        }

        return report

    # ==========================================================================
    # Engine Statistics
    # ==========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Return engine statistics.

        Returns:
            Dictionary with engine_id, version, check_count, and enabled frameworks.

        Example:
            >>> stats = engine.get_engine_stats()
            >>> stats["engine_id"]
            'eol_compliance_checker_engine'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "check_count": self._check_count,
            "enabled_frameworks": self._enabled_frameworks,
            "strict_mode": self._strict_mode,
        }

    # ==========================================================================
    # Private Helpers
    # ==========================================================================

    @staticmethod
    def _get_eu_diversion_target(reporting_year: int) -> Decimal:
        """
        Get the applicable EU Waste Framework Directive diversion target.

        Args:
            reporting_year: The reporting year.

        Returns:
            Diversion target as a percentage.
        """
        if isinstance(reporting_year, str):
            try:
                reporting_year = int(reporting_year)
            except (ValueError, TypeError):
                return Decimal("55")

        if reporting_year >= 2035:
            return EU_DIVERSION_TARGETS[2035]
        elif reporting_year >= 2030:
            return EU_DIVERSION_TARGETS[2030]
        else:
            return EU_DIVERSION_TARGETS[2025]


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
        'eol_compliance_checker_engine'
    """
    return ComplianceCheckerEngine.get_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "ComplianceFramework",
    "ComplianceStatus",
    "ComplianceSeverity",
    "DoubleCountingCategory",
    "BoundaryClassification",
    "TreatmentPathway",
    "WasteHierarchyLevel",
    "ComplianceFinding",
    "ComplianceCheckResult",
    "FrameworkCheckState",
    "FRAMEWORK_WEIGHTS",
    "ComplianceCheckerEngine",
    "get_compliance_checker",
]
