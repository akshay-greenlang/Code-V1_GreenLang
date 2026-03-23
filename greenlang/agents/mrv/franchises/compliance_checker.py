# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-027 Engine 6

This module implements regulatory compliance checking for Franchise emissions
(GHG Protocol Scope 3 Category 14) against 7 regulatory frameworks.
Category 14 is reported by the FRANCHISOR for emissions from the operation
of franchises not included in Scope 1 and Scope 2.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 14 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 Climate Change
4. CDP Climate Change Questionnaire (C6.5)
5. SBTi (Science Based Targets initiative)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 Emissions Standard

Category 14-Specific Compliance Rules:
- Consolidation approach consistency (financial control / equity share)
- Boundary validation (company-owned vs franchised)
- Data coverage threshold (minimum % of units with data)
- Franchise agreement type documentation
- Multi-tier franchise hierarchy verification
- Pro-rata for partial-year operations
- Franchise-specific preferred over average/spend-based
- Must report by franchise type

Double-Counting Prevention Rules (8 rules):
    DC-FRN-001: Company-owned units MUST be in Scope 1/2, NOT Cat 14
    DC-FRN-002: Do not double-count with Cat 13 (downstream leased assets)
    DC-FRN-003: Franchisee Scope 1/2 becomes franchisor Cat 14 -- boundary clarity
    DC-FRN-004: Multi-brand: if franchisee operates multiple brands, allocate by brand
    DC-FRN-005: Master franchise: do not count sub-franchisee twice
    DC-FRN-006: Transition units: company-owned converting to franchise (pro-rata)
    DC-FRN-007: Scope 2: grid electricity at franchise -- boundary with franchisor Scope 2
    DC-FRN-008: Supply chain: Cat 1 (purchased goods) vs Cat 14 (franchise ops) boundary

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_compliance(calculation_result, ["ghg_protocol", "cdp"])
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
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

ENGINE_ID: str = "gl_frn_compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# Minimum data coverage threshold (% of franchise units with primary data)
DEFAULT_MIN_DATA_COVERAGE: Decimal = Decimal("0.50")

# SBTi materiality threshold -- include Cat 14 if >= 40% of total Scope 3
SBTI_MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("40")

# SB 253 materiality threshold -- include if > 1% of total Scope 3
SB253_MATERIALITY_THRESHOLD_PCT: Decimal = Decimal("1")


# ==============================================================================
# ENUMS
# ==============================================================================


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks for franchise emissions."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class ComplianceStatus(str, Enum):
    """Compliance check status."""

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
    """Scope categories that could overlap with Category 14."""

    SCOPE_1 = "SCOPE_1"       # Company-owned franchise units
    SCOPE_2 = "SCOPE_2"       # Grid electricity at franchise
    CATEGORY_1 = "CATEGORY_1"  # Purchased goods & services
    CATEGORY_13 = "CATEGORY_13"  # Downstream leased assets


class BoundaryClassification(str, Enum):
    """Boundary classification for a franchise unit."""

    CATEGORY_14 = "CATEGORY_14"   # Franchise (correct for Cat 14)
    SCOPE_1_2 = "SCOPE_1_2"       # Company-owned (belongs in Scope 1/2)
    CATEGORY_13 = "CATEGORY_13"   # Downstream leased asset
    EXCLUDED = "EXCLUDED"         # Not in scope


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""

    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"
    EQUITY_SHARE = "equity_share"


class FranchiseAgreementType(str, Enum):
    """Types of franchise agreements."""

    SINGLE_UNIT = "single_unit"
    MULTI_UNIT = "multi_unit"
    MASTER_FRANCHISE = "master_franchise"
    AREA_DEVELOPMENT = "area_development"
    CONVERSION = "conversion"
    SUB_FRANCHISE = "sub_franchise"


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
    """Result of compliance check for a single framework."""

    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal
    findings: List[Dict[str, Any]]
    recommendations: List[str]


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

        Returns:
            Decimal score clamped to 0-100 range.
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
        """Convert accumulated state to a ComplianceCheckResult."""
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

# Map from config/string framework names to (enum, method_name)
_FRAMEWORK_DISPATCH: Dict[str, Tuple[ComplianceFramework, str]] = {
    "GHG_PROTOCOL_SCOPE3": (ComplianceFramework.GHG_PROTOCOL, "check_ghg_protocol"),
    "GHG_PROTOCOL": (ComplianceFramework.GHG_PROTOCOL, "check_ghg_protocol"),
    "ghg_protocol": (ComplianceFramework.GHG_PROTOCOL, "check_ghg_protocol"),
    "ISO_14064": (ComplianceFramework.ISO_14064, "check_iso_14064"),
    "iso_14064": (ComplianceFramework.ISO_14064, "check_iso_14064"),
    "CSRD_ESRS_E1": (ComplianceFramework.CSRD_ESRS, "check_csrd_esrs"),
    "CSRD_ESRS": (ComplianceFramework.CSRD_ESRS, "check_csrd_esrs"),
    "csrd_esrs": (ComplianceFramework.CSRD_ESRS, "check_csrd_esrs"),
    "CDP": (ComplianceFramework.CDP, "check_cdp"),
    "cdp": (ComplianceFramework.CDP, "check_cdp"),
    "SBTI": (ComplianceFramework.SBTI, "check_sbti"),
    "sbti": (ComplianceFramework.SBTI, "check_sbti"),
    "SB_253": (ComplianceFramework.SB_253, "check_sb253"),
    "sb_253": (ComplianceFramework.SB_253, "check_sb253"),
    "GRI": (ComplianceFramework.GRI, "check_gri"),
    "gri": (ComplianceFramework.GRI, "check_gri"),
}


# ==============================================================================
# REQUIRED DISCLOSURES PER FRAMEWORK
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_co2e",
        "calculation_method",
        "ef_sources",
        "franchise_type_breakdown",
        "consolidation_approach",
        "data_coverage",
        "exclusions",
    ],
    ComplianceFramework.ISO_14064: [
        "total_co2e",
        "methodology",
        "uncertainty_analysis",
        "base_year",
        "reporting_period",
        "boundary_description",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_co2e",
        "methodology",
        "targets",
        "franchise_type_breakdown",
        "actions",
        "xbrl_tags",
    ],
    ComplianceFramework.CDP: [
        "total_co2e",
        "data_quality_score",
        "engagement_metrics",
        "verification_status",
        "franchise_type_breakdown",
    ],
    ComplianceFramework.SBTI: [
        "total_co2e",
        "target_coverage",
        "progress_tracking",
        "franchise_engagement_target",
    ],
    ComplianceFramework.SB_253: [
        "total_co2e",
        "methodology",
        "assurance_opinion",
        "materiality_assessment",
    ],
    ComplianceFramework.GRI: [
        "total_co2e",
        "gases_included",
        "base_year",
        "standards_used",
        "ef_sources",
    ],
}


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for Franchise emissions (Category 14).

    Validates calculation results against 7 regulatory frameworks with
    Category 14-specific rules including franchise boundary enforcement,
    consolidation approach consistency, double-counting prevention (8 rules),
    data coverage thresholds, and multi-tier franchise hierarchy verification.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _enabled_frameworks: Set of enabled compliance frameworks
        _check_count: Running count of compliance checks performed
        _min_data_coverage: Minimum fraction of units requiring primary data

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> results = engine.check_compliance(calc_result, ["ghg_protocol"])
        >>> summary = engine.get_compliance_summary(results)
        >>> print(f"Overall: {summary['overall_status']}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with default configuration."""
        self._enabled_frameworks: List[str] = [
            "ghg_protocol", "iso_14064", "csrd_esrs",
            "cdp", "sbti", "sb_253", "gri",
        ]
        self._check_count: int = 0
        self._min_data_coverage: Decimal = DEFAULT_MIN_DATA_COVERAGE
        self._strict_mode: bool = False

        logger.info(
            "ComplianceCheckerEngine initialized: version=%s, "
            "frameworks=%d",
            ENGINE_VERSION,
            len(self._enabled_frameworks),
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

    def check_compliance(
        self,
        result: dict,
        frameworks: Optional[List[str]] = None,
    ) -> List[ComplianceCheckResult]:
        """
        Run compliance checks against specified or all enabled frameworks.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked.

        Args:
            result: Calculation result dictionary containing total_co2e,
                franchise_type_breakdown, consolidation_approach,
                data_coverage, etc.
            frameworks: Optional list of framework identifiers to check.
                If None, checks all enabled frameworks.

        Returns:
            List of ComplianceCheckResult, one per framework checked.

        Example:
            >>> results = engine.check_compliance(calc_result, ["ghg_protocol"])
            >>> results[0].status
            <ComplianceStatus.PASS: 'PASS'>
        """
        start_time = time.monotonic()
        logger.info("Running compliance checks for franchise emissions")

        fw_list = frameworks if frameworks is not None else self._enabled_frameworks
        all_results: List[ComplianceCheckResult] = []

        for framework_name in fw_list:
            dispatch = _FRAMEWORK_DISPATCH.get(framework_name)
            if dispatch is None:
                logger.warning(
                    "Unknown framework '%s', skipping", framework_name,
                )
                continue

            framework_enum, method_name = dispatch
            try:
                check_method = getattr(self, method_name)
                check_result = check_method(result)
                all_results.append(check_result)

                logger.info(
                    "%s compliance: %s (score: %s)",
                    framework_enum.value,
                    check_result.status.value,
                    check_result.score,
                )

            except Exception as e:
                logger.error(
                    "Error checking %s compliance: %s",
                    framework_name, str(e), exc_info=True,
                )
                all_results.append(ComplianceCheckResult(
                    framework=framework_enum,
                    status=ComplianceStatus.FAIL,
                    score=Decimal("0"),
                    findings=[{
                        "rule_code": "CHECK_ERROR",
                        "description": f"Compliance check failed: {str(e)}",
                        "severity": ComplianceSeverity.CRITICAL.value,
                        "status": ComplianceStatus.FAIL.value,
                    }],
                    recommendations=[
                        "Resolve the compliance check error and rerun."
                    ],
                ))

        duration = time.monotonic() - start_time
        self._check_count += 1

        logger.info(
            "All framework checks complete: %d frameworks, duration=%.4fs",
            len(all_results), duration,
        )

        return all_results

    def check_all_frameworks(
        self, result: dict,
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Run all enabled framework checks and return results keyed by name.

        Convenience method that returns a dictionary instead of a list.

        Args:
            result: Calculation result dictionary.

        Returns:
            Dictionary mapping framework name string to ComplianceCheckResult.

        Example:
            >>> all_results = engine.check_all_frameworks(calc_result)
            >>> ghg_result = all_results["ghg_protocol"]
        """
        check_results = self.check_compliance(result)
        return {cr.framework.value: cr for cr in check_results}

    # ==========================================================================
    # Framework: GHG Protocol Scope 3 (Category 14)
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 14).

        Category 14-specific GHG Protocol requirements:
            - Total emissions present and positive
            - Calculation method documented (franchise-specific preferred)
            - Emission factor sources documented
            - Franchise type breakdown provided
            - Consolidation approach documented and consistent
            - Data coverage meets minimum threshold
            - Exclusions documented
            - DQI score present
            - Double-counting rules validated

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

        # GHG-FRN-001: Total emissions present and positive
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GHG-FRN-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-FRN-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Ensure total_co2e is calculated and > 0.",
                regulation_reference="GHG Protocol Scope 3, Ch 14",
            )

        # GHG-FRN-002: Calculation method documented
        method = result.get("method") or result.get("calculation_method")
        if method:
            state.add_pass("GHG-FRN-002", "Calculation method documented")
            # Recommend franchise-specific over other methods
            method_str = str(method).lower()
            if method_str in ("spend_based", "spend"):
                state.add_warning(
                    "GHG-FRN-002a",
                    "Spend-based method used; franchise-specific data preferred",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Transition to franchise-specific data collection "
                        "for higher data quality and accuracy."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Table 14.1",
                )
        else:
            state.add_fail(
                "GHG-FRN-002",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used "
                    "(franchise-specific, average-data, spend-based, or hybrid)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 14.1",
            )

        # GHG-FRN-003: Emission factor sources documented
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GHG-FRN-003", "Emission factor sources documented")
        else:
            state.add_fail(
                "GHG-FRN-003",
                "Emission factor sources not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document all emission factor sources "
                    "(eGRID, IEA, EUI benchmarks, CBECS, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 14",
            )

        # GHG-FRN-004: Franchise type breakdown
        franchise_breakdown = (
            result.get("franchise_type_breakdown")
            or result.get("by_franchise_type")
            or result.get("breakdown_by_type")
        )
        if franchise_breakdown and isinstance(franchise_breakdown, dict) and len(franchise_breakdown) > 0:
            state.add_pass("GHG-FRN-004", "Franchise type breakdown provided")
        else:
            state.add_fail(
                "GHG-FRN-004",
                "Franchise type breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by franchise type "
                    "(e.g., quick-service restaurant, hotel, convenience store)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 14, Table 14.2",
            )

        # GHG-FRN-005: Consolidation approach documented
        consolidation = result.get("consolidation_approach")
        if consolidation:
            state.add_pass("GHG-FRN-005", "Consolidation approach documented")
        else:
            state.add_fail(
                "GHG-FRN-005",
                "Consolidation approach not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the consolidation approach used "
                    "(financial control, operational control, or equity share). "
                    "Under financial/operational control, franchise units are "
                    "Category 14. Under equity share, proportional emissions apply."
                ),
                regulation_reference="GHG Protocol Corporate Standard, Ch 3",
            )

        # GHG-FRN-006: Data coverage meets threshold
        self._check_data_coverage_state(result, state, "GHG-FRN-006")

        # GHG-FRN-007: Exclusions documented
        exclusions = result.get("exclusions")
        if exclusions is not None:
            state.add_pass("GHG-FRN-007", "Exclusions documented")
        else:
            state.add_warning(
                "GHG-FRN-007",
                "Exclusions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document any exclusions from Category 14 reporting "
                    "(e.g., franchise units below materiality threshold, "
                    "newly acquired units)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 14",
            )

        # GHG-FRN-008: DQI score present
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("GHG-FRN-008", "DQI score present")
        else:
            state.add_warning(
                "GHG-FRN-008",
                "Data quality indicator (DQI) score not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across 5 dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-FRN-009: Double-counting validation
        dc_violations = self._check_dc_rules(result)
        if not dc_violations:
            state.add_pass("GHG-FRN-009", "No double-counting violations detected")
        else:
            for violation in dc_violations:
                state.add_fail(
                    violation,
                    f"Double-counting violation: {violation}",
                    ComplianceSeverity.CRITICAL,
                    recommendation=self._get_dc_recommendation(violation),
                    regulation_reference="GHG Protocol Scope 3, Ch 14, Appendix A",
                )

        # GHG-FRN-010: Consolidation consistency check
        consistency_issues = self._check_consolidation_approach(result)
        if not consistency_issues:
            state.add_pass("GHG-FRN-010", "Consolidation approach applied consistently")
        else:
            for issue in consistency_issues:
                state.add_fail(
                    "GHG-FRN-010",
                    issue,
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Apply the same consolidation approach across all "
                        "franchise units consistently."
                    ),
                    regulation_reference="GHG Protocol Corporate Standard, Ch 3",
                )

        return state.to_result()

    # ==========================================================================
    # Framework: ISO 14064-1:2018
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018.

        Checks:
            - total_co2e present
            - Uncertainty analysis present
            - Base year documented
            - Methodology described
            - Reporting period defined
            - Boundary description documented

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

        # ISO-FRN-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("ISO-FRN-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-FRN-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total CO2e emissions.",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-FRN-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-FRN-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-FRN-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis "
                    "(Monte Carlo, analytical, or IPCC Tier 2). "
                    "Franchise data may have higher uncertainty due to "
                    "estimation and extrapolation."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-FRN-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-FRN-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-FRN-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison "
                    "and trend analysis. Consider network growth adjustments."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-FRN-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-FRN-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-FRN-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, data sources, calculation approach, "
                    "and how franchise-level data was collected/estimated."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-FRN-005: Reporting period defined
        reporting_period = result.get("reporting_period") or result.get("period")
        if reporting_period:
            state.add_pass("ISO-FRN-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-FRN-005",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation="Specify the reporting period (e.g., 2025, 2025-Q3).",
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-FRN-006: Boundary description
        boundary = (
            result.get("boundary_description")
            or result.get("organizational_boundary")
            or result.get("consolidation_approach")
        )
        if boundary:
            state.add_pass("ISO-FRN-006", "Organizational boundary described")
        else:
            state.add_fail(
                "ISO-FRN-006",
                "Organizational boundary not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the organizational boundary decisions for "
                    "franchise operations, including which units are included "
                    "and the rationale for the consolidation approach."
                ),
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
            - total_co2e reported
            - Methodology description
            - Targets documented
            - Franchise type breakdown present
            - Actions described
            - Value chain coverage documented

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

        # CSRD-FRN-001: Total emissions reported
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CSRD-FRN-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-FRN-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Scope 3 Category 14 emissions.",
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-FRN-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-FRN-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-FRN-002",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for franchise "
                    "emissions as required by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-FRN-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-FRN-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-FRN-003",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for franchise "
                    "operations (e.g., energy efficiency upgrades, "
                    "renewable energy procurement, equipment standards)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-FRN-004: Franchise type breakdown present
        franchise_breakdown = (
            result.get("franchise_type_breakdown")
            or result.get("by_franchise_type")
            or result.get("breakdown_by_type")
        )
        if franchise_breakdown and isinstance(franchise_breakdown, dict) and len(franchise_breakdown) > 0:
            state.add_pass("CSRD-FRN-004", "Franchise type breakdown present")
        else:
            state.add_fail(
                "CSRD-FRN-004",
                "Franchise type breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by franchise type "
                    "for CSRD value chain reporting."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-FRN-005: Actions described
        actions = result.get("actions") or result.get("reduction_actions")
        if actions:
            state.add_pass("CSRD-FRN-005", "Reduction actions described")
        else:
            state.add_warning(
                "CSRD-FRN-005",
                "Reduction actions not described",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Describe actions taken or planned to reduce franchise "
                    "emissions (e.g., energy audits, equipment upgrade programs, "
                    "renewable energy targets for franchisees)."
                ),
                regulation_reference="ESRS E1-3, para 29",
            )

        # CSRD-FRN-006: Value chain coverage
        value_chain_coverage = (
            result.get("value_chain_coverage")
            or result.get("data_coverage")
        )
        if value_chain_coverage is not None:
            state.add_pass("CSRD-FRN-006", "Value chain coverage documented")
        else:
            state.add_warning(
                "CSRD-FRN-006",
                "Value chain coverage not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the percentage of franchise network covered "
                    "by primary vs estimated data."
                ),
                regulation_reference="ESRS E1-6, para 52",
            )

        # CSRD-FRN-007: XBRL tagging readiness
        xbrl_tags = result.get("xbrl_tags") or result.get("xbrl_ready")
        if xbrl_tags:
            state.add_pass("CSRD-FRN-007", "XBRL tagging addressed")
        else:
            state.add_warning(
                "CSRD-FRN-007",
                "XBRL tagging not addressed",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Prepare data for XBRL-tagged digital reporting "
                    "as required by CSRD."
                ),
                regulation_reference="CSRD Art. 29d",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CDP Climate Change
    # ==========================================================================

    def check_cdp(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change Questionnaire.

        CDP Requirements for Category 14 Franchises:
            - C6.5: Category 14 emissions figure
            - Data quality disclosure
            - Engagement metrics for franchisees
            - Verification status documented

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

        # CDP-FRN-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CDP-FRN-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CDP-FRN-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 14 emissions to CDP.",
                regulation_reference="CDP CC C6.5",
            )

        # CDP-FRN-002: Data quality disclosure
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("CDP-FRN-002", "Data quality score disclosed")
        else:
            state.add_fail(
                "CDP-FRN-002",
                "Data quality score not disclosed",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Disclose data quality score for franchise emissions. "
                    "CDP evaluates data quality and methodology transparency."
                ),
                regulation_reference="CDP CC C6.5",
            )

        # CDP-FRN-003: Engagement metrics
        engagement = (
            result.get("engagement_metrics")
            or result.get("franchisee_engagement")
            or result.get("engagement_rate")
        )
        if engagement is not None:
            state.add_pass("CDP-FRN-003", "Franchisee engagement metrics present")
        else:
            state.add_warning(
                "CDP-FRN-003",
                "Franchisee engagement metrics not provided",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Report franchisee engagement metrics "
                    "(% of franchisees providing data, engagement programs, "
                    "data collection campaigns)."
                ),
                regulation_reference="CDP CC C6.5, C12.1",
            )

        # CDP-FRN-004: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-FRN-004", "Verification status documented")
        else:
            state.add_warning(
                "CDP-FRN-004",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 14 franchise emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module 10, Q10.1",
            )

        # CDP-FRN-005: Franchise type breakdown
        franchise_breakdown = (
            result.get("franchise_type_breakdown")
            or result.get("by_franchise_type")
        )
        if franchise_breakdown and isinstance(franchise_breakdown, dict):
            state.add_pass("CDP-FRN-005", "Franchise type breakdown provided")
        else:
            state.add_warning(
                "CDP-FRN-005",
                "Franchise type breakdown not provided for CDP",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide emissions breakdown by franchise type for "
                    "granular CDP reporting."
                ),
                regulation_reference="CDP CC C6.5",
            )

        # CDP-FRN-006: Methodology documented
        method = result.get("method") or result.get("calculation_method")
        if method:
            state.add_pass("CDP-FRN-006", "Calculation methodology documented")
        else:
            state.add_fail(
                "CDP-FRN-006",
                "Calculation methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology used for "
                    "franchise emissions (franchise-specific, average-data, "
                    "spend-based, or hybrid)."
                ),
                regulation_reference="CDP CC C6.5",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SBTi
    # ==========================================================================

    def check_sbti(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative.

        SBTi requirements for Category 14:
            - Include if >= 40% of total Scope 3
            - Franchise engagement target
            - FLAG (Forests, Land and Agriculture) if applicable
            - Progress tracking present

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

        # SBTI-FRN-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SBTI-FRN-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-FRN-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate total Category 14 emissions for SBTi target boundary.",
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-FRN-002: Materiality assessment (>= 40% threshold)
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat14_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                if cat14_pct >= SBTI_MATERIALITY_THRESHOLD_PCT:
                    state.add_pass(
                        "SBTI-FRN-002",
                        f"Category 14 is material ({cat14_pct:.1f}% of Scope 3, >= 40%)",
                    )
                else:
                    state.add_warning(
                        "SBTI-FRN-002",
                        f"Category 14 below SBTi threshold ({cat14_pct:.1f}% of Scope 3, < 40%)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 14 may not need a separate SBTi target "
                            "if below 40% materiality threshold, but must still "
                            "be included in the screening."
                        ),
                        regulation_reference="SBTi Criteria v5.1, C20",
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-FRN-002",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-FRN-002",
                "Total Scope 3 emissions not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 14 materiality."
                ),
            )

        # SBTI-FRN-003: Franchise engagement target
        engagement_target = (
            result.get("franchise_engagement_target")
            or result.get("engagement_target")
            or result.get("sbti_engagement_target")
        )
        if engagement_target is not None:
            state.add_pass("SBTI-FRN-003", "Franchise engagement target documented")
        else:
            state.add_warning(
                "SBTI-FRN-003",
                "Franchise engagement target not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document a franchise engagement target "
                    "(e.g., % of franchisees with approved science-based targets "
                    "within 5 years). SBTi allows supplier/customer engagement "
                    "targets for Scope 3 categories."
                ),
                regulation_reference="SBTi Criteria v5.1, C20-C22",
            )

        # SBTI-FRN-004: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-FRN-004", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-FRN-004",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 14 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-FRN-005: FLAG assessment (if applicable)
        has_flag = result.get("flag_applicable") or result.get("flag_assessment")
        franchise_types = result.get("franchise_types", [])
        flag_relevant = any(
            ft in ("agriculture", "food_production", "forestry")
            for ft in (
                [str(ft).lower() for ft in franchise_types]
                if franchise_types
                else []
            )
        )
        if flag_relevant:
            if has_flag:
                state.add_pass("SBTI-FRN-005", "FLAG assessment documented")
            else:
                state.add_warning(
                    "SBTI-FRN-005",
                    "FLAG assessment not documented but may be applicable",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Assess whether SBTi FLAG guidance applies to "
                        "franchise operations involving food/agriculture/forestry."
                    ),
                    regulation_reference="SBTi FLAG Guidance v2",
                )
        else:
            state.add_pass("SBTI-FRN-005", "FLAG assessment not applicable")

        return state.to_result()

    # ==========================================================================
    # Framework: SB 253
    # ==========================================================================

    def check_sb253(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Checks:
            - Total CO2e present
            - Methodology documented
            - Assurance opinion available
            - Materiality > 1% threshold

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for SB 253.

        Example:
            >>> res = engine.check_sb253(calc_result)
            >>> res.framework.value
            'sb_253'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # SB253-FRN-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SB253-FRN-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-FRN-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 14 emissions for SB 253 compliance.",
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-FRN-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-FRN-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-FRN-002",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-FRN-003: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-FRN-003", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-FRN-003",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-FRN-004: Materiality > 1% threshold
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat14_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                if cat14_pct > SB253_MATERIALITY_THRESHOLD_PCT:
                    state.add_pass(
                        "SB253-FRN-004",
                        f"Category 14 exceeds 1% materiality ({cat14_pct:.2f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SB253-FRN-004",
                        f"Category 14 below 1% materiality ({cat14_pct:.2f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 14 is below 1% of total Scope 3. "
                            "Consider whether separate reporting is warranted."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SB253-FRN-004",
                    "Could not calculate materiality",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SB253-FRN-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 14 materiality against 1% threshold."
                ),
            )

        # SB253-FRN-005: Data coverage for assurance readiness
        self._check_data_coverage_state(result, state, "SB253-FRN-005")

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
            - Source of emission factors
            - Methodological transparency

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

        # GRI-FRN-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GRI-FRN-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-FRN-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 14 emissions in metric tonnes CO2e.",
                regulation_reference="GRI 305-3",
            )

        # GRI-FRN-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-FRN-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-FRN-002",
                "Gases included not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, HFCs, or CO2e aggregate). "
                    "For franchises, include electricity (CO2) and refrigerants (HFCs)."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-FRN-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-FRN-003", "Base year present")
        else:
            state.add_warning(
                "GRI-FRN-003",
                "Base year not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting. "
                    "Consider network growth adjustments."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-FRN-004: Standards referenced
        standards = (
            result.get("standards_used")
            or result.get("standards")
            or result.get("framework_references")
        )
        if standards:
            state.add_pass("GRI-FRN-004", "Standards referenced")
        else:
            state.add_warning(
                "GRI-FRN-004",
                "Standards used not referenced",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Reference the standards and methodologies used "
                    "(e.g., GHG Protocol Scope 3 Category 14, CBECS, ENERGY STAR)."
                ),
                regulation_reference="GRI 305-3(e)",
            )

        # GRI-FRN-005: Source of emission factors
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-FRN-005", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-FRN-005",
                "Emission factor sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., eGRID 2024, IEA 2024, CBECS 2024)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        # GRI-FRN-006: Methodological transparency
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("GRI-FRN-006", "Methodological transparency maintained")
        else:
            state.add_warning(
                "GRI-FRN-006",
                "Methodology not documented for GRI transparency",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the methodology, assumptions, and data sources "
                    "used for franchise emissions calculations."
                ),
                regulation_reference="GRI 305-3(e)",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention (8 rules)
    # ==========================================================================

    def _check_dc_rules(self, result: dict) -> List[str]:
        """
        Validate 8 double-counting prevention rules for franchise emissions.

        Rules:
            DC-FRN-001: Company-owned units MUST be in Scope 1/2, NOT Cat 14
            DC-FRN-002: No double-count with Cat 13 (downstream leased assets)
            DC-FRN-003: Franchisee Scope 1/2 -> franchisor Cat 14 boundary clarity
            DC-FRN-004: Multi-brand allocation by brand
            DC-FRN-005: Master franchise: no sub-franchisee double count
            DC-FRN-006: Transition units: pro-rata for company-to-franchise
            DC-FRN-007: Grid electricity boundary with franchisor Scope 2
            DC-FRN-008: Cat 1 vs Cat 14 boundary for supply chain

        Args:
            result: Calculation result dictionary containing franchise unit
                details and boundary information.

        Returns:
            List of violation rule codes (empty if no violations).

        Example:
            >>> violations = engine._check_dc_rules(calc_result)
            >>> len(violations)
            0
        """
        violations: List[str] = []

        # DC-FRN-001: Company-owned units must NOT be in Cat 14
        units = result.get("units") or result.get("franchise_units") or []
        for idx, unit in enumerate(units):
            ownership = str(unit.get("ownership_type", "")).lower()
            if ownership in ("company_owned", "company", "corporate"):
                violations.append("DC-FRN-001")
                logger.warning(
                    "DC-FRN-001: Unit %d is company-owned but included in Cat 14",
                    idx,
                )
                break  # Report once

        # DC-FRN-002: Check for overlap with Cat 13 (downstream leased assets)
        also_in_cat13 = result.get("reported_in_cat13") or result.get("cat13_overlap")
        if also_in_cat13:
            violations.append("DC-FRN-002")
            logger.warning(
                "DC-FRN-002: Overlap detected with Category 13 (downstream leased assets)",
            )

        # DC-FRN-003: Boundary clarity -- franchisee Scope 1/2 = franchisor Cat 14
        boundary_documented = result.get("boundary_documented") or result.get("boundary_description")
        if not boundary_documented and units:
            violations.append("DC-FRN-003")
            logger.warning(
                "DC-FRN-003: Boundary between franchisee Scope 1/2 and franchisor Cat 14 not documented",
            )

        # DC-FRN-004: Multi-brand allocation
        multi_brand_units = [
            u for u in units
            if u.get("brands") and len(u.get("brands", [])) > 1
        ]
        if multi_brand_units:
            for unit in multi_brand_units:
                allocation = unit.get("brand_allocation")
                if not allocation:
                    violations.append("DC-FRN-004")
                    logger.warning(
                        "DC-FRN-004: Multi-brand unit found without brand allocation",
                    )
                    break  # Report once

        # DC-FRN-005: Master franchise hierarchy
        master_units = [
            u for u in units
            if str(u.get("agreement_type", "")).lower() == "master_franchise"
        ]
        sub_units = [
            u for u in units
            if str(u.get("agreement_type", "")).lower() == "sub_franchise"
        ]
        if master_units and sub_units:
            # Check for overlap -- sub-franchise should not be counted independently
            # if already counted under master
            master_ids = {u.get("master_franchise_id") for u in sub_units if u.get("master_franchise_id")}
            counted_masters = {u.get("unit_id") for u in master_units}
            if master_ids & counted_masters:
                # Sub-franchisees reference masters that are also being counted
                sub_included = result.get("sub_franchise_included_in_master", True)
                if not sub_included:
                    violations.append("DC-FRN-005")
                    logger.warning(
                        "DC-FRN-005: Sub-franchisees counted separately from master franchise",
                    )

        # DC-FRN-006: Transition units (company-owned -> franchise)
        transition_units = [
            u for u in units
            if u.get("is_transition") or u.get("transition_date")
        ]
        for unit in transition_units:
            pro_rata = unit.get("pro_rata_applied") or unit.get("pro_rata_fraction")
            if not pro_rata:
                violations.append("DC-FRN-006")
                logger.warning(
                    "DC-FRN-006: Transition unit without pro-rata allocation",
                )
                break

        # DC-FRN-007: Grid electricity boundary
        electricity_boundary = result.get("electricity_boundary_documented")
        scope2_overlap = result.get("scope2_franchise_electricity_overlap")
        if scope2_overlap:
            violations.append("DC-FRN-007")
            logger.warning(
                "DC-FRN-007: Franchise electricity also counted in franchisor Scope 2",
            )

        # DC-FRN-008: Cat 1 vs Cat 14 supply chain boundary
        cat1_overlap = result.get("cat1_overlap") or result.get("purchased_goods_overlap")
        if cat1_overlap:
            violations.append("DC-FRN-008")
            logger.warning(
                "DC-FRN-008: Overlap between Cat 1 (purchased goods) and Cat 14 (franchise ops)",
            )

        logger.info(
            "Double-counting check: %d violations found",
            len(violations),
        )

        return violations

    def check_double_counting(
        self, units: list,
    ) -> List[dict]:
        """
        Validate double-counting prevention rules at the unit level.

        Provides detailed per-unit findings for all 8 DC rules.

        Args:
            units: List of franchise unit dictionaries.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and affected unit indices.

        Example:
            >>> findings = engine.check_double_counting(units)
            >>> len(findings)
            0
        """
        findings: List[dict] = []

        for idx, unit in enumerate(units):
            # DC-FRN-001: Company-owned
            ownership = str(unit.get("ownership_type", "")).lower()
            if ownership in ("company_owned", "company", "corporate"):
                findings.append({
                    "rule_code": "DC-FRN-001",
                    "description": (
                        f"Unit {idx}: Company-owned unit detected. "
                        "Must be in Scope 1/2, not Category 14."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "unit_index": idx,
                    "recommendation": (
                        "Exclude company-owned/corporate units from Category 14. "
                        "Report their emissions under Scope 1 (energy) and Scope 2 (electricity)."
                    ),
                })

            # DC-FRN-002: Leased asset overlap
            also_leased = unit.get("reported_in_cat13", False)
            if also_leased:
                findings.append({
                    "rule_code": "DC-FRN-002",
                    "description": (
                        f"Unit {idx}: Also reported in Category 13 "
                        "(downstream leased assets). Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "unit_index": idx,
                    "recommendation": (
                        "Report each property in only one category. "
                        "If the unit is a franchise, use Category 14. "
                        "If it is a leased asset, use Category 13."
                    ),
                })

            # DC-FRN-004: Multi-brand without allocation
            brands = unit.get("brands", [])
            if isinstance(brands, list) and len(brands) > 1:
                allocation = unit.get("brand_allocation")
                if not allocation:
                    findings.append({
                        "rule_code": "DC-FRN-004",
                        "description": (
                            f"Unit {idx}: Operates {len(brands)} brands "
                            "but no brand-level allocation provided."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "unit_index": idx,
                        "recommendation": (
                            "Allocate emissions by brand using revenue share, "
                            "floor area, or operating hours as allocation basis."
                        ),
                    })

            # DC-FRN-005: Sub-franchise counted separately
            agreement = str(unit.get("agreement_type", "")).lower()
            if agreement == "sub_franchise":
                master_id = unit.get("master_franchise_id")
                if not master_id:
                    findings.append({
                        "rule_code": "DC-FRN-005",
                        "description": (
                            f"Unit {idx}: Sub-franchise without master_franchise_id. "
                            "Risk of double-counting at master and sub level."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "unit_index": idx,
                        "recommendation": (
                            "Link sub-franchise to master franchise and ensure "
                            "emissions are counted at only one level in the hierarchy."
                        ),
                    })

            # DC-FRN-006: Transition without pro-rata
            if unit.get("is_transition") or unit.get("transition_date"):
                pro_rata = unit.get("pro_rata_applied") or unit.get("pro_rata_fraction")
                if not pro_rata:
                    findings.append({
                        "rule_code": "DC-FRN-006",
                        "description": (
                            f"Unit {idx}: Transition unit (company-to-franchise) "
                            "without pro-rata allocation."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "unit_index": idx,
                        "recommendation": (
                            "Apply pro-rata allocation for transition units. "
                            "Emissions before transition date -> Scope 1/2. "
                            "Emissions after transition date -> Category 14."
                        ),
                    })

            # DC-FRN-007: Electricity boundary
            electricity_in_scope2 = unit.get("electricity_in_franchisor_scope2", False)
            if electricity_in_scope2:
                findings.append({
                    "rule_code": "DC-FRN-007",
                    "description": (
                        f"Unit {idx}: Franchise electricity also in franchisor Scope 2. "
                        "Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "unit_index": idx,
                    "recommendation": (
                        "Franchise electricity consumption should be in Category 14 "
                        "(franchisor perspective) OR Scope 2 of the franchisee. "
                        "Do not count in franchisor's own Scope 2."
                    ),
                })

            # DC-FRN-008: Supply chain overlap
            supply_chain_overlap = unit.get("supply_chain_in_cat1", False)
            if supply_chain_overlap:
                findings.append({
                    "rule_code": "DC-FRN-008",
                    "description": (
                        f"Unit {idx}: Franchise operations supply chain also "
                        "counted in Category 1 (Purchased Goods & Services)."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "unit_index": idx,
                    "recommendation": (
                        "Category 14 covers franchise OPERATIONS emissions "
                        "(energy, refrigerants). Supply chain emissions of "
                        "goods sold TO franchisees belong in Category 1."
                    ),
                })

        logger.info(
            "Per-unit double-counting check: %d units, %d findings",
            len(units), len(findings),
        )

        return findings

    # ==========================================================================
    # Boundary Validation
    # ==========================================================================

    def _validate_boundary(self, result: dict) -> List[str]:
        """
        Validate franchise boundary classification.

        Ensures all units are correctly classified as Category 14
        (franchise) vs Scope 1/2 (company-owned) vs Category 13
        (downstream leased assets).

        Args:
            result: Calculation result dictionary.

        Returns:
            List of boundary issue descriptions.

        Example:
            >>> issues = engine._validate_boundary(calc_result)
            >>> len(issues)
            0
        """
        issues: List[str] = []
        units = result.get("units") or result.get("franchise_units") or []

        for idx, unit in enumerate(units):
            classification = self._classify_unit_boundary(unit)
            if classification != BoundaryClassification.CATEGORY_14.value:
                issues.append(
                    f"Unit {idx}: classified as {classification}, "
                    f"should not be in Category 14"
                )

        # Check consolidation approach documented
        consolidation = result.get("consolidation_approach")
        if not consolidation:
            issues.append(
                "Consolidation approach not documented. Required to determine "
                "which units belong in Scope 1/2 vs Category 14."
            )

        return issues

    def _classify_unit_boundary(self, unit: dict) -> str:
        """
        Classify a franchise unit into the correct GHG reporting boundary.

        Args:
            unit: Franchise unit dictionary.

        Returns:
            BoundaryClassification value string.
        """
        ownership = str(unit.get("ownership_type", "")).lower()

        # Company-owned -> Scope 1/2
        if ownership in ("company_owned", "company", "corporate", "wholly_owned"):
            return BoundaryClassification.SCOPE_1_2.value

        # Leased asset -> Category 13
        if ownership in ("leased", "downstream_lease"):
            return BoundaryClassification.CATEGORY_13.value

        # Franchise -> Category 14
        if ownership in (
            "franchise", "franchised", "franchisee",
            "master_franchise", "sub_franchise", "area_development",
        ):
            return BoundaryClassification.CATEGORY_14.value

        # Default: classify as Category 14 if it has a franchise agreement
        agreement = str(unit.get("agreement_type", "")).lower()
        if agreement in (
            "single_unit", "multi_unit", "master_franchise",
            "area_development", "conversion", "sub_franchise",
        ):
            return BoundaryClassification.CATEGORY_14.value

        return BoundaryClassification.EXCLUDED.value

    # ==========================================================================
    # Data Coverage
    # ==========================================================================

    def _check_data_coverage(self, result: dict) -> ComplianceCheckResult:
        """
        Check data coverage meets minimum threshold.

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for data coverage check.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)
        self._check_data_coverage_state(result, state, "DATA-COV")
        return state.to_result()

    def _check_data_coverage_state(
        self, result: dict, state: FrameworkCheckState, rule_prefix: str,
    ) -> None:
        """
        Check data coverage and add findings to an existing state.

        Validates that the fraction of franchise units with primary data
        meets the minimum threshold (default 50%).

        Args:
            result: Calculation result dictionary.
            state: Framework check state to add findings to.
            rule_prefix: Rule code prefix for findings.
        """
        data_coverage = result.get("data_coverage")
        total_units = result.get("total_units") or result.get("total_franchise_units")
        units_with_data = result.get("units_with_data") or result.get("units_with_primary_data")

        if data_coverage is not None:
            coverage_dec = Decimal(str(data_coverage))
            if coverage_dec >= self._min_data_coverage:
                state.add_pass(
                    rule_prefix,
                    f"Data coverage {coverage_dec:.0%} meets threshold {self._min_data_coverage:.0%}",
                )
            else:
                state.add_warning(
                    rule_prefix,
                    f"Data coverage {coverage_dec:.0%} below threshold {self._min_data_coverage:.0%}",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        f"Increase data coverage to at least {self._min_data_coverage:.0%} "
                        "of franchise units with primary (franchise-specific) data. "
                        "Engage franchisees in data collection programs."
                    ),
                )
        elif total_units and units_with_data:
            try:
                coverage_dec = Decimal(str(units_with_data)) / Decimal(str(total_units))
                if coverage_dec >= self._min_data_coverage:
                    state.add_pass(
                        rule_prefix,
                        f"Data coverage {coverage_dec:.0%} ({units_with_data}/{total_units} units)",
                    )
                else:
                    state.add_warning(
                        rule_prefix,
                        f"Data coverage {coverage_dec:.0%} ({units_with_data}/{total_units} units) "
                        f"below threshold {self._min_data_coverage:.0%}",
                        ComplianceSeverity.HIGH,
                        recommendation=(
                            "Increase primary data collection from franchisees. "
                            "Consider tiered data collection (largest units first)."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    rule_prefix,
                    "Could not calculate data coverage",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                rule_prefix,
                "Data coverage information not provided",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Report data coverage: number of franchise units with "
                    "primary data vs total units in network."
                ),
            )

    # ==========================================================================
    # Consolidation Approach
    # ==========================================================================

    def _check_consolidation_approach(self, result: dict) -> List[str]:
        """
        Check that consolidation approach is applied consistently.

        Validates:
        - Approach is documented
        - Same approach is used across all units
        - Company-owned units excluded under control approaches

        Args:
            result: Calculation result dictionary.

        Returns:
            List of consistency issue descriptions.

        Example:
            >>> issues = engine._check_consolidation_approach(calc_result)
            >>> len(issues)
            0
        """
        issues: List[str] = []

        consolidation = result.get("consolidation_approach")
        if not consolidation:
            return issues  # Already flagged in framework checks

        approach = str(consolidation).lower()

        # Validate known approach
        valid_approaches = {
            "financial_control", "operational_control", "equity_share",
        }
        if approach not in valid_approaches:
            issues.append(
                f"Unknown consolidation approach '{consolidation}'. "
                f"Must be one of: {', '.join(sorted(valid_approaches))}."
            )
            return issues

        # Under financial or operational control, company-owned units
        # are in Scope 1/2, NOT Category 14
        units = result.get("units") or result.get("franchise_units") or []
        if approach in ("financial_control", "operational_control"):
            for idx, unit in enumerate(units):
                ownership = str(unit.get("ownership_type", "")).lower()
                if ownership in ("company_owned", "company", "corporate"):
                    issues.append(
                        f"Unit {idx} is company-owned but included in Category 14. "
                        f"Under {approach}, company-owned units belong in Scope 1/2."
                    )

        # Under equity share, check proportional allocation
        if approach == "equity_share":
            for idx, unit in enumerate(units):
                equity_pct = unit.get("equity_share_pct") or unit.get("equity_share")
                if equity_pct is None:
                    issues.append(
                        f"Unit {idx}: equity share percentage not specified. "
                        "Required for equity share consolidation approach."
                    )

        # Check all units use the same approach
        unit_approaches = {
            str(u.get("consolidation_approach", approach)).lower()
            for u in units
            if u.get("consolidation_approach")
        }
        if len(unit_approaches) > 1:
            issues.append(
                f"Inconsistent consolidation approaches across units: "
                f"{', '.join(sorted(unit_approaches))}. "
                "Must use same approach for all units."
            )

        return issues

    # ==========================================================================
    # Recommendations
    # ==========================================================================

    def _generate_recommendations(self, violations: List[str]) -> List[str]:
        """
        Generate actionable recommendations from violation codes.

        Args:
            violations: List of violation rule codes.

        Returns:
            List of recommendation strings.

        Example:
            >>> recs = engine._generate_recommendations(["DC-FRN-001"])
            >>> len(recs)
            1
        """
        recommendations: List[str] = []
        seen: Set[str] = set()

        for code in violations:
            rec = self._get_dc_recommendation(code)
            if rec and rec not in seen:
                recommendations.append(rec)
                seen.add(rec)

        return recommendations

    @staticmethod
    def _get_dc_recommendation(code: str) -> str:
        """
        Get recommendation text for a double-counting rule code.

        Args:
            code: DC rule code (e.g., 'DC-FRN-001').

        Returns:
            Recommendation string.
        """
        _dc_recommendations: Dict[str, str] = {
            "DC-FRN-001": (
                "Exclude company-owned/corporate units from Category 14. "
                "Report their emissions under Scope 1 (stationary/mobile combustion) "
                "and Scope 2 (purchased electricity/heat/cooling)."
            ),
            "DC-FRN-002": (
                "Do not report the same property in both Category 13 (downstream "
                "leased assets) and Category 14 (franchises). Choose one category "
                "based on the contractual arrangement."
            ),
            "DC-FRN-003": (
                "Document the boundary between franchisee Scope 1/2 emissions and "
                "the franchisor's Category 14 reporting. Franchisee's Scope 1/2 "
                "becomes the franchisor's Category 14."
            ),
            "DC-FRN-004": (
                "For multi-brand franchisees, allocate emissions by brand using "
                "revenue share, floor area, or operating hours as the allocation basis."
            ),
            "DC-FRN-005": (
                "In master franchise hierarchies, count emissions at only one level. "
                "Either the master franchise reports aggregate data, or each "
                "sub-franchise reports individually, but not both."
            ),
            "DC-FRN-006": (
                "For units transitioning from company-owned to franchise, apply "
                "pro-rata allocation based on transition date. Before transition: "
                "Scope 1/2. After transition: Category 14."
            ),
            "DC-FRN-007": (
                "Franchise electricity consumption should appear in Category 14 "
                "(franchisor perspective), NOT in the franchisor's own Scope 2. "
                "It is the franchisee's Scope 2."
            ),
            "DC-FRN-008": (
                "Category 14 covers franchise OPERATIONS emissions (energy, refrigerants, "
                "mobile combustion). Goods/services supplied TO franchisees belong in "
                "Category 1 (Purchased Goods & Services), not Category 14."
            ),
        }
        return _dc_recommendations.get(code, f"Review and resolve: {code}")

    # ==========================================================================
    # Summary & Recommendations
    # ==========================================================================

    def get_compliance_summary(
        self, results: List[ComplianceCheckResult],
    ) -> Dict[str, Any]:
        """
        Summarize all framework results with overall score and recommendations.

        Calculates a weighted average compliance score across all checked
        frameworks and aggregates findings by severity.

        Args:
            results: List of ComplianceCheckResult from check_compliance().

        Returns:
            Summary dictionary with overall_score, overall_status,
            framework_scores, findings by severity, and recommendations.

        Example:
            >>> summary = engine.get_compliance_summary(results)
            >>> summary["overall_score"]
            85.5
            >>> summary["overall_status"]
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
                "total_findings": 0,
                "recommendations": [],
            }

        # Calculate weighted overall score
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        framework_scores: Dict[str, Dict[str, Any]] = {}

        for check_result in results:
            fw_enum = check_result.framework
            weight = FRAMEWORK_WEIGHTS.get(fw_enum, Decimal("1.00"))

            weighted_sum += check_result.score * weight
            total_weight += weight

            framework_scores[fw_enum.value] = {
                "score": float(check_result.score),
                "status": check_result.status.value,
                "findings_count": len(check_result.findings),
                "weight": float(weight),
            }

        overall_score = Decimal("0")
        if total_weight > 0:
            overall_score = (weighted_sum / total_weight).quantize(
                _QUANT_2DP, rounding=ROUNDING,
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

        for check_result in results:
            framework_name = check_result.framework.value
            for finding in check_result.findings:
                finding_with_fw = {"framework": framework_name, **finding}
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

    def get_overall_compliance_score(
        self, results: List[ComplianceCheckResult],
    ) -> Decimal:
        """
        Calculate the overall weighted compliance score.

        Args:
            results: List of ComplianceCheckResult.

        Returns:
            Weighted average score as Decimal (0-100).

        Example:
            >>> score = engine.get_overall_compliance_score(results)
            >>> float(score)
            92.5
        """
        if not results:
            return Decimal("0.00")

        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for check_result in results:
            weight = FRAMEWORK_WEIGHTS.get(check_result.framework, Decimal("1.00"))
            weighted_sum += check_result.score * weight
            total_weight += weight

        if total_weight == 0:
            return Decimal("0.00")

        return (weighted_sum / total_weight).quantize(_QUANT_2DP, rounding=ROUNDING)

    # ==========================================================================
    # Engine Stats
    # ==========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Return engine statistics.

        Returns:
            Dictionary with engine_id, version, check_count, and config.

        Example:
            >>> stats = engine.get_engine_stats()
            >>> stats["engine_id"]
            'gl_frn_compliance_checker_engine'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "check_count": self._check_count,
            "enabled_frameworks": self._enabled_frameworks,
            "strict_mode": self._strict_mode,
            "min_data_coverage": float(self._min_data_coverage),
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
        'gl_frn_compliance_checker_engine'
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
    "ConsolidationApproach",
    "FranchiseAgreementType",
    "ComplianceFinding",
    "ComplianceCheckResult",
    "FrameworkCheckState",
    "FRAMEWORK_WEIGHTS",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "ComplianceCheckerEngine",
    "get_compliance_checker",
]
