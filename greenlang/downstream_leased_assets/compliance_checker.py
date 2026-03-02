# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-026 Engine 6

This module implements regulatory compliance checking for Downstream Leased Assets
emissions (GHG Protocol Scope 3 Category 13) against 7 regulatory frameworks.

Category 13 covers emissions from the operation of assets OWNED by the reporting
company and LEASED TO other entities. The reporter is the LESSOR. This is the
mirror of Category 8 (Upstream Leased Assets) where the reporter is the lessee.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 13 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 Climate Change (with E1-5 energy performance, EPC ratings)
4. CDP Climate Change Questionnaire (C6.5)
5. SBTi (Science Based Targets initiative)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 Emissions Standard

Category 13-Specific Compliance Rules:
- Operational control boundary (lessor vs lessee control distinction)
- Consolidation approach validation (financial control vs equity share)
- Tenant data coverage assessment
- Asset breakdown by type (building, vehicle, equipment, IT)
- Allocation method disclosure (floor area, headcount, revenue, custom)
- Data quality scoring (5-dimension DQI)
- Double-counting prevention (8 rules: DC-DLA-001 through DC-DLA-008)

Double-Counting Prevention Rules:
    DC-DLA-001: Operational control -- if lessor has it, Scope 1/2 not Cat 13
    DC-DLA-002: Same asset cannot be in both Cat 8 (lessee) and Cat 13 (lessor)
    DC-DLA-003: Finance lease -- lessee may have operational control
    DC-DLA-004: Sub-leasing -- intermediate lessor not asset owner
    DC-DLA-005: Common area energy -- proportional allocation, no double-count
    DC-DLA-006: Scope 2 boundary -- grid electricity in leased buildings
    DC-DLA-007: REIT properties -- operational control vs Cat 13 distinction
    DC-DLA-008: Fleet -- do not double-count with Cat 11 (use of sold products)

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
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
AGENT_ID: str = "GL-MRV-S3-013"
AGENT_COMPONENT: str = "AGENT-MRV-026"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
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


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class ComplianceFramework(str, Enum):
    """Supported regulatory compliance frameworks."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories and scopes that could overlap with Category 13."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    CATEGORY_8 = "CATEGORY_8"   # Upstream leased assets (lessee side)
    CATEGORY_11 = "CATEGORY_11"  # Use of sold products (fleet overlap)
    CATEGORY_3 = "CATEGORY_3"   # Fuel & energy activities (WTT)


class BoundaryClassification(str, Enum):
    """Boundary classification for leased assets."""

    CATEGORY_13 = "CATEGORY_13"  # Downstream leased assets (correct)
    SCOPE_1 = "SCOPE_1"         # Lessor has operational control
    SCOPE_2 = "SCOPE_2"         # Purchased electricity for common areas
    CATEGORY_8 = "CATEGORY_8"   # Lessee perspective (wrong side)
    CATEGORY_11 = "CATEGORY_11"  # Use of sold products
    EXCLUDED = "EXCLUDED"        # Not in scope


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class LeaseType(str, Enum):
    """Lease classification types."""

    OPERATING_LEASE = "operating_lease"
    FINANCE_LEASE = "finance_lease"
    CAPITAL_LEASE = "capital_lease"
    SUBLEASE = "sublease"


class AssetCategory(str, Enum):
    """Leased asset category types."""

    BUILDING = "building"
    VEHICLE = "vehicle"
    EQUIPMENT = "equipment"
    IT_ASSET = "it_asset"


class AllocationMethod(str, Enum):
    """Allocation methods for shared spaces and multi-tenant assets."""

    FLOOR_AREA = "floor_area"
    HEADCOUNT = "headcount"
    REVENUE = "revenue"
    METERED = "metered"
    EQUAL_SHARE = "equal_share"
    CUSTOM = "custom"


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
    """Result of a single framework compliance check."""

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


# ==============================================================================
# FRAMEWORK REQUIRED DISCLOSURES
# ==============================================================================

FRAMEWORK_REQUIRED_DISCLOSURES: Dict[str, List[str]] = {
    "ghg_protocol": [
        "total_co2e",
        "asset_breakdown",
        "allocation_method",
        "consolidation_approach",
        "boundary_validation",
        "method_justification",
        "dqi_score",
        "tenant_data_coverage",
    ],
    "iso_14064": [
        "total_co2e",
        "uncertainty",
        "base_year",
        "methodology",
        "reporting_period",
        "verification",
    ],
    "csrd_esrs": [
        "total_co2e",
        "methodology",
        "targets",
        "asset_breakdown",
        "energy_performance",
        "epc_ratings",
        "green_lease_clauses",
        "tenant_engagement",
    ],
    "cdp": [
        "total_co2e",
        "asset_breakdown",
        "data_quality",
        "methodology",
        "verification",
    ],
    "sbti": [
        "total_co2e",
        "coverage_67pct",
        "base_year",
        "progress",
        "materiality",
        "reduction_pathway",
    ],
    "sb_253": [
        "total_co2e",
        "methodology",
        "assurance",
        "materiality_1pct",
        "completeness",
    ],
    "gri": [
        "total_co2e",
        "gases_included",
        "base_year",
        "ef_sources",
    ],
}


# ==============================================================================
# BUILDING ENERGY PERFORMANCE BENCHMARKS (kWh/m2/year by building type)
# ==============================================================================

BUILDING_EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "office": {
        "excellent": Decimal("100"),
        "good": Decimal("150"),
        "typical": Decimal("200"),
        "poor": Decimal("300"),
    },
    "retail": {
        "excellent": Decimal("120"),
        "good": Decimal("180"),
        "typical": Decimal("250"),
        "poor": Decimal("380"),
    },
    "warehouse": {
        "excellent": Decimal("50"),
        "good": Decimal("80"),
        "typical": Decimal("120"),
        "poor": Decimal("200"),
    },
    "residential": {
        "excellent": Decimal("80"),
        "good": Decimal("120"),
        "typical": Decimal("160"),
        "poor": Decimal("250"),
    },
    "hotel": {
        "excellent": Decimal("150"),
        "good": Decimal("220"),
        "typical": Decimal("300"),
        "poor": Decimal("450"),
    },
    "healthcare": {
        "excellent": Decimal("200"),
        "good": Decimal("300"),
        "typical": Decimal("400"),
        "poor": Decimal("550"),
    },
    "education": {
        "excellent": Decimal("90"),
        "good": Decimal("140"),
        "typical": Decimal("190"),
        "poor": Decimal("280"),
    },
    "data_center": {
        "excellent": Decimal("500"),
        "good": Decimal("800"),
        "typical": Decimal("1200"),
        "poor": Decimal("2000"),
    },
}


# ==============================================================================
# EPC RATING THRESHOLDS (Energy Performance Certificate)
# ==============================================================================

EPC_RATING_SCORES: Dict[str, int] = {
    "A": 100,
    "B": 85,
    "C": 70,
    "D": 55,
    "E": 40,
    "F": 25,
    "G": 10,
}


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    Compliance checker for Downstream Leased Assets emissions (Category 13).

    Validates calculation results against 7 regulatory frameworks with
    Category 13-specific rules including operational control boundary,
    consolidation approach validation, tenant data coverage, double-counting
    prevention, and building energy performance assessment.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _enabled_frameworks: List of enabled compliance framework identifiers
        _check_count: Running count of compliance checks performed
        _strict_mode: Whether to enforce strict compliance rules
        _materiality_threshold: Materiality threshold for SBTi/SB 253

    Example:
        >>> engine = ComplianceCheckerEngine.get_instance()
        >>> results = engine.check_all_frameworks(calc_result)
        >>> summary = engine.get_compliance_summary(results)
        >>> print(f"Overall: {summary['overall_status']}")
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with default configuration."""
        self._enabled_frameworks: List[str] = [
            "GHG_PROTOCOL_SCOPE3",
            "ISO_14064",
            "CSRD_ESRS_E1",
            "CDP",
            "SBTI",
            "SB_253",
            "GRI",
        ]
        self._strict_mode: bool = False
        self._materiality_threshold: Decimal = Decimal("0.01")
        self._check_count: int = 0

        # Attempt to load config from the downstream_leased_assets config module
        try:
            from greenlang.downstream_leased_assets.config import get_config
            config = get_config()
            if config is not None and hasattr(config, "compliance"):
                comp = config.compliance
                if hasattr(comp, "get_frameworks"):
                    self._enabled_frameworks = comp.get_frameworks()
                if hasattr(comp, "strict_mode"):
                    self._strict_mode = comp.strict_mode
                if hasattr(comp, "materiality_threshold"):
                    self._materiality_threshold = comp.materiality_threshold
        except (ImportError, AttributeError, Exception):
            pass

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
                asset_breakdown, allocation_method, consolidation_approach,
                tenant_data_coverage, etc.

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

    # ==========================================================================
    # Framework: GHG Protocol Scope 3 (Category 13)
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 13).

        Checks (8 rules):
            GHG-DLA-001: Total CO2e present and positive
            GHG-DLA-002: Asset breakdown by category documented
            GHG-DLA-003: Allocation method documented
            GHG-DLA-004: Consolidation approach specified
            GHG-DLA-005: Boundary validation (operational control check)
            GHG-DLA-006: Method justification documented
            GHG-DLA-007: Data quality indicator (DQI) score present
            GHG-DLA-008: Tenant data coverage percentage reported

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

        # GHG-DLA-001: Total emissions present and positive
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GHG-DLA-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-DLA-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Ensure total_co2e is calculated and > 0.",
                regulation_reference="GHG Protocol Scope 3, Ch 6.13",
            )

        # GHG-DLA-002: Asset breakdown documented
        asset_breakdown = (
            result.get("asset_breakdown")
            or result.get("by_asset_category")
            or result.get("by_asset_type")
        )
        if asset_breakdown and isinstance(asset_breakdown, dict) and len(asset_breakdown) > 0:
            state.add_pass("GHG-DLA-002", "Asset breakdown by category documented")
        else:
            state.add_fail(
                "GHG-DLA-002",
                "Asset breakdown by category not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by asset category "
                    "(building, vehicle, equipment, IT asset)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 6.13",
            )

        # GHG-DLA-003: Allocation method documented
        allocation_method = (
            result.get("allocation_method")
            or result.get("allocation_approach")
        )
        if allocation_method:
            state.add_pass("GHG-DLA-003", "Allocation method documented")
        else:
            state.add_fail(
                "GHG-DLA-003",
                "Allocation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the allocation method used for multi-tenant "
                    "assets (floor_area, headcount, revenue, metered, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 6.13",
            )

        # GHG-DLA-004: Consolidation approach specified
        consolidation = (
            result.get("consolidation_approach")
            or result.get("consolidation_method")
        )
        if consolidation:
            valid_approaches = {
                "operational_control", "financial_control", "equity_share",
            }
            if str(consolidation).lower() in valid_approaches:
                state.add_pass(
                    "GHG-DLA-004",
                    "Consolidation approach specified and valid",
                )
            else:
                state.add_warning(
                    "GHG-DLA-004",
                    f"Consolidation approach '{consolidation}' not recognized",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Use one of: operational_control, financial_control, "
                        "equity_share."
                    ),
                    regulation_reference="GHG Protocol Corporate Standard, Ch 3",
                )
        else:
            state.add_fail(
                "GHG-DLA-004",
                "Consolidation approach not specified",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Specify the consolidation approach: operational_control, "
                    "financial_control, or equity_share."
                ),
                regulation_reference="GHG Protocol Corporate Standard, Ch 3",
            )

        # GHG-DLA-005: Boundary validation (operational control check)
        boundary_validated = result.get("boundary_validated", False)
        operational_control_check = result.get("operational_control_check")
        if boundary_validated or operational_control_check:
            state.add_pass(
                "GHG-DLA-005",
                "Boundary validation performed (operational control checked)",
            )
        else:
            state.add_fail(
                "GHG-DLA-005",
                "Boundary validation not performed",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Validate that the reporter does NOT have operational "
                    "control over leased assets. If the lessor operates the "
                    "assets, emissions belong in Scope 1/2, not Cat 13."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 6.13",
            )

        # GHG-DLA-006: Method justification documented
        method_justification = (
            result.get("method_justification")
            or result.get("method")
            or result.get("calculation_method")
        )
        if method_justification:
            state.add_pass("GHG-DLA-006", "Calculation method justified")
        else:
            state.add_fail(
                "GHG-DLA-006",
                "Calculation method not justified",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document and justify the calculation method: "
                    "asset-specific, average-data, spend-based, or hybrid."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 6.13",
            )

        # GHG-DLA-007: DQI score present
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("GHG-DLA-007", "DQI score present")
        else:
            state.add_warning(
                "GHG-DLA-007",
                "Data quality indicator (DQI) score not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across 5 dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-DLA-008: Tenant data coverage
        tenant_coverage = (
            result.get("tenant_data_coverage")
            or result.get("tenant_coverage_pct")
        )
        if tenant_coverage is not None:
            coverage_pct = Decimal(str(tenant_coverage))
            if coverage_pct >= Decimal("80"):
                state.add_pass(
                    "GHG-DLA-008",
                    f"Tenant data coverage is {coverage_pct}% (>= 80%)",
                )
            elif coverage_pct >= Decimal("50"):
                state.add_warning(
                    "GHG-DLA-008",
                    f"Tenant data coverage is {coverage_pct}% (below 80%)",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Improve tenant data coverage to at least 80% through "
                        "metering, tenant questionnaires, or green lease clauses."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 6.13",
                )
            else:
                state.add_fail(
                    "GHG-DLA-008",
                    f"Tenant data coverage is only {coverage_pct}% (below 50%)",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Tenant data coverage below 50% significantly reduces "
                        "data quality. Use benchmarks for assets without tenant "
                        "data and implement engagement programs."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 6.13",
                )
        else:
            state.add_warning(
                "GHG-DLA-008",
                "Tenant data coverage percentage not reported",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Report the percentage of leased assets with actual "
                    "tenant energy data vs. estimated/benchmark data."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 6.13",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: ISO 14064
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018.

        Checks (6 rules):
            ISO-DLA-001: Total CO2e present
            ISO-DLA-002: Uncertainty analysis present
            ISO-DLA-003: Base year documented
            ISO-DLA-004: Methodology described
            ISO-DLA-005: Reporting period defined
            ISO-DLA-006: Verification status documented

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

        # ISO-DLA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("ISO-DLA-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-DLA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Calculate and report total CO2e emissions.",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-DLA-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-DLA-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-DLA-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis "
                    "(Monte Carlo, analytical, or IPCC Tier 2). "
                    "Downstream leased assets typically have +/-30-50% "
                    "uncertainty due to tenant data gaps."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-DLA-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-DLA-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-DLA-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison "
                    "and trend analysis. Consider asset portfolio "
                    "changes when recalculating the base year."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-DLA-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-DLA-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-DLA-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, data sources, and calculation approach "
                    "(asset-specific, average-data, spend-based, or hybrid)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-DLA-005: Reporting period defined
        reporting_period = (
            result.get("reporting_period")
            or result.get("period")
        )
        if reporting_period:
            state.add_pass("ISO-DLA-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-DLA-005",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Specify the reporting period (e.g., 2025, 2025-Q3). "
                    "Align with lease terms and occupancy periods."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-DLA-006: Verification status documented
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("ISO-DLA-006", "Verification status documented")
        else:
            state.add_warning(
                "ISO-DLA-006",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the third-party verification status. "
                    "ISO 14064-3 provides guidance on verification."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 10",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CSRD / ESRS E1
    # ==========================================================================

    def check_csrd_esrs(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CSRD ESRS E1 Climate Change.

        Checks (8 rules):
            CSRD-DLA-001: Total CO2e by category
            CSRD-DLA-002: Methodology description
            CSRD-DLA-003: Targets documented
            CSRD-DLA-004: Asset breakdown present
            CSRD-DLA-005: ESRS E1-5 energy performance of buildings
            CSRD-DLA-006: EPC ratings reported
            CSRD-DLA-007: Green lease clauses documented
            CSRD-DLA-008: Tenant engagement described

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

        # CSRD-DLA-001: Total emissions by category
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CSRD-DLA-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-DLA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Scope 3 Category 13 emissions.",
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-DLA-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-DLA-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-DLA-002",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for downstream "
                    "leased assets emissions as required by ESRS E1."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-DLA-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-DLA-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-DLA-003",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for leased portfolio "
                    "(e.g., improve building efficiency, EPC upgrades, "
                    "green lease rollout)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-DLA-004: Asset breakdown present
        asset_breakdown = (
            result.get("asset_breakdown")
            or result.get("by_asset_category")
        )
        if asset_breakdown and isinstance(asset_breakdown, dict) and len(asset_breakdown) > 0:
            state.add_pass("CSRD-DLA-004", "Asset breakdown present")
        else:
            state.add_fail(
                "CSRD-DLA-004",
                "Asset breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by asset category "
                    "(building, vehicle, equipment, IT asset)."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-DLA-005: ESRS E1-5 energy performance of buildings
        energy_performance = (
            result.get("energy_performance")
            or result.get("building_energy_intensity")
            or result.get("eui_benchmarks")
        )
        if energy_performance:
            state.add_pass(
                "CSRD-DLA-005",
                "Energy performance of buildings disclosed (ESRS E1-5)",
            )
        else:
            building_assets = self._has_building_assets(result)
            if building_assets:
                state.add_fail(
                    "CSRD-DLA-005",
                    "Energy performance of buildings not disclosed",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Disclose energy performance (EUI in kWh/m2/year) "
                        "for leased buildings as required by ESRS E1-5. "
                        "Include energy source breakdown."
                    ),
                    regulation_reference="ESRS E1-5, para 37-41",
                )
            else:
                state.add_pass(
                    "CSRD-DLA-005",
                    "No building assets; energy performance disclosure N/A",
                )

        # CSRD-DLA-006: EPC ratings reported
        epc_ratings = (
            result.get("epc_ratings")
            or result.get("epc_distribution")
            or result.get("energy_certificates")
        )
        if epc_ratings:
            state.add_pass("CSRD-DLA-006", "EPC ratings reported")
        else:
            building_assets = self._has_building_assets(result)
            if building_assets:
                state.add_warning(
                    "CSRD-DLA-006",
                    "EPC ratings not reported for leased buildings",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Report EPC (Energy Performance Certificate) rating "
                        "distribution for the leased building portfolio. "
                        "Include % of portfolio by EPC grade (A through G)."
                    ),
                    regulation_reference="ESRS E1-5, para 40",
                )
            else:
                state.add_pass(
                    "CSRD-DLA-006",
                    "No building assets; EPC rating disclosure N/A",
                )

        # CSRD-DLA-007: Green lease clauses documented
        green_lease = (
            result.get("green_lease_clauses")
            or result.get("green_lease_coverage")
            or result.get("has_green_leases")
        )
        if green_lease is not None:
            state.add_pass("CSRD-DLA-007", "Green lease clauses documented")
        else:
            state.add_warning(
                "CSRD-DLA-007",
                "Green lease clauses not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document green lease clause coverage across the "
                    "portfolio. Green leases include provisions for energy "
                    "data sharing, efficiency improvements, and sustainability "
                    "commitments between lessor and tenant."
                ),
                regulation_reference="ESRS E1-3, para 29",
            )

        # CSRD-DLA-008: Tenant engagement described
        tenant_engagement = (
            result.get("tenant_engagement")
            or result.get("tenant_engagement_program")
        )
        if tenant_engagement:
            state.add_pass("CSRD-DLA-008", "Tenant engagement described")
        else:
            state.add_warning(
                "CSRD-DLA-008",
                "Tenant engagement not described",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Describe tenant engagement programs for emission "
                    "reduction: data sharing incentives, energy efficiency "
                    "workshops, sustainability committees, etc."
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

        Checks (5 rules):
            CDP-DLA-001: C6.5 total emissions present
            CDP-DLA-002: Asset breakdown documented
            CDP-DLA-003: Data quality assessment
            CDP-DLA-004: Methodology documented
            CDP-DLA-005: Verification status documented

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

        # CDP-DLA-001: Total emissions present (C6.5)
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CDP-DLA-001", "Total CO2e reported (C6.5)")
        else:
            state.add_fail(
                "CDP-DLA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 13 emissions to CDP. "
                    "This maps to C6.5 (Scope 3 emissions by category)."
                ),
                regulation_reference="CDP CC Module 6, Q6.5",
            )

        # CDP-DLA-002: Asset breakdown documented
        asset_breakdown = (
            result.get("asset_breakdown")
            or result.get("by_asset_category")
        )
        if asset_breakdown and isinstance(asset_breakdown, dict) and len(asset_breakdown) > 0:
            state.add_pass("CDP-DLA-002", "Asset breakdown documented")
        else:
            state.add_fail(
                "CDP-DLA-002",
                "Asset breakdown not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by asset type "
                    "(building, vehicle, equipment, IT) for CDP disclosure."
                ),
                regulation_reference="CDP CC Module 6, Q6.5",
            )

        # CDP-DLA-003: Data quality assessment
        dqi_score = (
            result.get("dqi_score")
            or result.get("data_quality_score")
            or result.get("data_quality")
        )
        if dqi_score is not None:
            state.add_pass("CDP-DLA-003", "Data quality assessment present")
        else:
            state.add_warning(
                "CDP-DLA-003",
                "Data quality assessment not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Provide a data quality assessment for Category 13. "
                    "CDP scores data quality, which affects scoring."
                ),
                regulation_reference="CDP CC Module 6, Q6.5",
            )

        # CDP-DLA-004: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CDP-DLA-004", "Methodology documented")
        else:
            state.add_fail(
                "CDP-DLA-004",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology. "
                    "CDP requires disclosure of the method used "
                    "(asset-specific, average-data, spend-based)."
                ),
                regulation_reference="CDP CC Module 6, Q6.5",
            )

        # CDP-DLA-005: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-DLA-005", "Verification status documented")
        else:
            state.add_warning(
                "CDP-DLA-005",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether Category 13 emissions have been "
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

        Checks (6 rules):
            SBTI-DLA-001: Total emissions present
            SBTI-DLA-002: 67% coverage threshold
            SBTI-DLA-003: Base year documented
            SBTI-DLA-004: Progress tracking present
            SBTI-DLA-005: Materiality assessment
            SBTI-DLA-006: Reduction pathway defined

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

        # SBTI-DLA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SBTI-DLA-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-DLA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Calculate total Category 13 emissions for SBTi target boundary."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-DLA-002: 67% coverage threshold
        target_coverage = (
            result.get("target_coverage")
            or result.get("sbti_coverage")
            or result.get("portfolio_coverage_pct")
        )
        if target_coverage is not None:
            coverage = Decimal(str(target_coverage))
            if coverage >= Decimal("67"):
                state.add_pass(
                    "SBTI-DLA-002",
                    f"Target coverage {coverage}% meets 67% threshold",
                )
            else:
                state.add_warning(
                    "SBTI-DLA-002",
                    f"Target coverage {coverage}% below 67% threshold",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "SBTi requires 67% of Scope 3 emissions to be "
                        "covered by targets. Increase coverage or justify "
                        "exclusion of Category 13."
                    ),
                    regulation_reference="SBTi Criteria v5.1, C20",
                )
        else:
            state.add_warning(
                "SBTI-DLA-002",
                "SBTi target coverage not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document what percentage of Scope 3 emissions is covered "
                    "by SBTi targets (minimum 67% required)."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-DLA-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("SBTI-DLA-003", "Base year documented")
        else:
            state.add_fail(
                "SBTI-DLA-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for SBTi target setting. "
                    "Must be within 2 years of target submission."
                ),
                regulation_reference="SBTi Criteria v5.1, C4",
            )

        # SBTI-DLA-004: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-DLA-004", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-DLA-004",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 13 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-DLA-005: Materiality assessment
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat13_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                threshold_pct = self._materiality_threshold * Decimal("100")
                if cat13_pct >= threshold_pct:
                    state.add_pass(
                        "SBTI-DLA-005",
                        f"Category 13 is material ({cat13_pct:.1f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SBTI-DLA-005",
                        f"Category 13 below materiality threshold "
                        f"({cat13_pct:.1f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 13 may not need to be included in SBTi "
                            "target boundary if below materiality threshold."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-DLA-005",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-DLA-005",
                "Total Scope 3 emissions not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 13 materiality."
                ),
            )

        # SBTI-DLA-006: Reduction pathway defined
        reduction_pathway = (
            result.get("reduction_pathway")
            or result.get("sbti_pathway")
            or result.get("decarbonization_plan")
        )
        if reduction_pathway:
            state.add_pass("SBTI-DLA-006", "Reduction pathway defined")
        else:
            state.add_warning(
                "SBTI-DLA-006",
                "Reduction pathway not defined",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Define a reduction pathway for Category 13: "
                    "building retrofits, green lease adoption, EPC "
                    "upgrade targets, fleet electrification for leased vehicles."
                ),
                regulation_reference="SBTi Criteria v5.1, C11",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SB 253
    # ==========================================================================

    def check_sb_253(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        Checks (5 rules):
            SB253-DLA-001: Total CO2e present
            SB253-DLA-002: Methodology documented
            SB253-DLA-003: Assurance opinion available
            SB253-DLA-004: Materiality > 1% threshold
            SB253-DLA-005: Completeness check

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

        # SB253-DLA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SB253-DLA-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-DLA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation="Report total Category 13 emissions for SB 253 compliance.",
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-DLA-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-DLA-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-DLA-002",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-DLA-003: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-DLA-003", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-DLA-003",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-DLA-004: Materiality > 1% threshold
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat13_pct = (
                    Decimal(str(total_co2e))
                    / Decimal(str(total_scope3))
                    * Decimal("100")
                )
                if cat13_pct > Decimal("1"):
                    state.add_pass(
                        "SB253-DLA-004",
                        f"Category 13 exceeds 1% materiality "
                        f"({cat13_pct:.2f}% of Scope 3)",
                    )
                else:
                    state.add_warning(
                        "SB253-DLA-004",
                        f"Category 13 below 1% materiality threshold "
                        f"({cat13_pct:.2f}% of Scope 3)",
                        ComplianceSeverity.LOW,
                        recommendation=(
                            "Category 13 is below 1% of total Scope 3. "
                            "Consider whether separate reporting is warranted."
                        ),
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SB253-DLA-004",
                    "Could not calculate materiality",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SB253-DLA-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 13 materiality against 1% threshold."
                ),
            )

        # SB253-DLA-005: Completeness check
        completeness = result.get("completeness_score") or result.get("completeness")
        if completeness is not None:
            completeness_val = Decimal(str(completeness))
            if completeness_val >= Decimal("90"):
                state.add_pass(
                    "SB253-DLA-005",
                    f"Completeness score {completeness_val}% (>= 90%)",
                )
            elif completeness_val >= Decimal("70"):
                state.add_warning(
                    "SB253-DLA-005",
                    f"Completeness score {completeness_val}% (between 70-90%)",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Improve completeness to at least 90%. "
                        "Ensure all leased asset categories are covered."
                    ),
                )
            else:
                state.add_fail(
                    "SB253-DLA-005",
                    f"Completeness score {completeness_val}% (below 70%)",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Completeness below 70% may not meet SB 253 requirements. "
                        "Review asset inventory and fill data gaps."
                    ),
                    regulation_reference="SB 253, Section 38532(c)",
                )
        else:
            state.add_warning(
                "SB253-DLA-005",
                "Completeness score not provided",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Assess and report the completeness of Category 13 "
                    "emissions coverage across the leased asset portfolio."
                ),
            )

        return state.to_result()

    # ==========================================================================
    # Framework: GRI 305
    # ==========================================================================

    def check_gri(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305 Emissions Standard.

        Checks (4 rules):
            GRI-DLA-001: Total CO2e in metric tonnes
            GRI-DLA-002: Gases included documented
            GRI-DLA-003: Base year present
            GRI-DLA-004: Emission factor sources documented

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

        # GRI-DLA-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GRI-DLA-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-DLA-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 13 emissions in metric tonnes CO2e."
                ),
                regulation_reference="GRI 305-3",
            )

        # GRI-DLA-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-DLA-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-DLA-002",
                "Gases included not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, HFCs, or CO2e aggregate). "
                    "For leased buildings, include electricity-related CO2 "
                    "and any on-site combustion gases."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-DLA-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-DLA-003", "Base year present")
        else:
            state.add_warning(
                "GRI-DLA-003",
                "Base year not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting. "
                    "Consider portfolio changes when recalculating base year."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-DLA-004: Source of emission factors
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-DLA-004", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-DLA-004",
                "Emission factor sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., eGRID 2024, IEA 2024, DEFRA 2024, CBECS)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention (8 Rules)
    # ==========================================================================

    def check_double_counting(
        self, assets: list
    ) -> List[dict]:
        """
        Validate 8 double-counting prevention rules for downstream leased assets.

        Rules:
            DC-DLA-001: Operational control -- if lessor has it, Scope 1/2 not Cat 13
            DC-DLA-002: Same asset cannot be in both Cat 8 (lessee) and Cat 13 (lessor)
            DC-DLA-003: Finance lease -- lessee may have operational control
            DC-DLA-004: Sub-leasing -- intermediate lessor not asset owner
            DC-DLA-005: Common area energy -- proportional allocation, no double-count
            DC-DLA-006: Scope 2 boundary -- grid electricity in leased buildings
            DC-DLA-007: REIT properties -- operational control vs Cat 13 distinction
            DC-DLA-008: Fleet -- do not double-count with Cat 11 (use of sold products)

        Args:
            assets: List of asset dictionaries, each containing fields like
                asset_id, asset_category, lease_type, operational_control,
                reported_in_scope1, reported_in_scope2, reported_in_cat8,
                reported_in_cat11, is_sublease, is_reit, common_area_included,
                grid_electricity_included, etc.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and affected asset indices.

        Example:
            >>> findings = engine.check_double_counting(assets)
            >>> len(findings)
            0  # No double-counting issues
        """
        findings: List[dict] = []

        for idx, asset in enumerate(assets):
            asset_id = asset.get("asset_id", f"asset_{idx}")

            # DC-DLA-001: Operational control check
            has_operational_control = asset.get("operational_control", False)
            if has_operational_control:
                findings.append({
                    "rule_code": "DC-DLA-001",
                    "description": (
                        f"Asset {asset_id}: Lessor has operational control. "
                        "Emissions should be reported under Scope 1/2, "
                        "not Category 13."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "asset_index": idx,
                    "asset_id": asset_id,
                    "recommendation": (
                        "If the lessor (reporter) operates the asset, "
                        "emissions belong in Scope 1 (direct) or Scope 2 "
                        "(purchased electricity), not Scope 3 Category 13."
                    ),
                })

            # DC-DLA-002: Cat 8 / Cat 13 mutual exclusivity
            also_in_cat8 = asset.get("reported_in_cat8", False)
            if also_in_cat8:
                findings.append({
                    "rule_code": "DC-DLA-002",
                    "description": (
                        f"Asset {asset_id}: Also reported in Category 8 "
                        "(upstream leased assets). An asset cannot be in "
                        "both Cat 8 (lessee) and Cat 13 (lessor)."
                    ),
                    "severity": ComplianceSeverity.CRITICAL.value,
                    "asset_index": idx,
                    "asset_id": asset_id,
                    "recommendation": (
                        "Ensure each leased asset appears in only one "
                        "category: Cat 8 if reporter is lessee, "
                        "Cat 13 if reporter is lessor."
                    ),
                })

            # DC-DLA-003: Finance lease -- lessee typically has operational control
            lease_type = asset.get("lease_type", "").lower()
            if lease_type in ("finance_lease", "capital_lease"):
                lessee_has_control = asset.get(
                    "lessee_has_operational_control", True
                )
                if lessee_has_control:
                    findings.append({
                        "rule_code": "DC-DLA-003",
                        "description": (
                            f"Asset {asset_id}: Finance/capital lease where "
                            "lessee has operational control. Verify correct "
                            "boundary classification."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "asset_index": idx,
                        "asset_id": asset_id,
                        "recommendation": (
                            "Under finance/capital lease, the lessee typically "
                            "has operational control. If the lessee operates "
                            "the asset, the lessee reports in Scope 1/2. "
                            "The lessor may report under Cat 13 only if "
                            "the lessee does not have operational control."
                        ),
                    })

            # DC-DLA-004: Sub-leasing check
            is_sublease = asset.get("is_sublease", False)
            if is_sublease:
                is_original_owner = asset.get("is_original_owner", False)
                if not is_original_owner:
                    findings.append({
                        "rule_code": "DC-DLA-004",
                        "description": (
                            f"Asset {asset_id}: Sub-lease detected and reporter "
                            "is not the original asset owner. Intermediate "
                            "lessors should not report under Cat 13."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "asset_index": idx,
                        "asset_id": asset_id,
                        "recommendation": (
                            "Only the original asset owner (primary lessor) "
                            "reports under Category 13. Sub-lessors should "
                            "report under Category 8 (upstream leased assets)."
                        ),
                    })

            # DC-DLA-005: Common area energy double-counting
            common_area_included = asset.get("common_area_included", False)
            common_area_in_scope2 = asset.get("common_area_in_scope2", False)
            if common_area_included and common_area_in_scope2:
                findings.append({
                    "rule_code": "DC-DLA-005",
                    "description": (
                        f"Asset {asset_id}: Common area energy included in "
                        "both Category 13 and Scope 2. Proportional allocation "
                        "required to prevent double-counting."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "asset_index": idx,
                    "asset_id": asset_id,
                    "recommendation": (
                        "Allocate common area energy proportionally between "
                        "Scope 2 (lessor share) and Category 13 (tenant share). "
                        "Use floor area or metered data for allocation."
                    ),
                })

            # DC-DLA-006: Scope 2 boundary for grid electricity
            grid_electricity_included = asset.get(
                "grid_electricity_included", False
            )
            grid_in_scope2 = asset.get("grid_in_scope2", False)
            if grid_electricity_included and grid_in_scope2:
                findings.append({
                    "rule_code": "DC-DLA-006",
                    "description": (
                        f"Asset {asset_id}: Grid electricity reported in "
                        "both Category 13 and Scope 2. Ensure correct "
                        "boundary assignment."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "asset_index": idx,
                    "asset_id": asset_id,
                    "recommendation": (
                        "Grid electricity purchased by the lessor for leased "
                        "buildings goes in Scope 2. Electricity purchased "
                        "by the tenant goes in Cat 13 (from lessor perspective). "
                        "Do not count the same electricity in both."
                    ),
                })

            # DC-DLA-007: REIT properties
            is_reit = asset.get("is_reit", False)
            if is_reit:
                reit_operational_control = asset.get(
                    "reit_operational_control", False
                )
                if reit_operational_control:
                    findings.append({
                        "rule_code": "DC-DLA-007",
                        "description": (
                            f"Asset {asset_id}: REIT property where REIT has "
                            "operational control. Should be Scope 1/2 for "
                            "the REIT, not Category 13."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "asset_index": idx,
                        "asset_id": asset_id,
                        "recommendation": (
                            "If the REIT operates the building (manages "
                            "operations, controls energy procurement), "
                            "report in Scope 1/2. Only use Cat 13 for "
                            "properties where the REIT is a passive lessor "
                            "and tenants control operations."
                        ),
                    })

            # DC-DLA-008: Fleet double-count with Cat 11
            asset_category = asset.get("asset_category", "").lower()
            reported_in_cat11 = asset.get("reported_in_cat11", False)
            if asset_category == "vehicle" and reported_in_cat11:
                findings.append({
                    "rule_code": "DC-DLA-008",
                    "description": (
                        f"Asset {asset_id}: Leased vehicle also reported in "
                        "Category 11 (use of sold products). Double-counting risk."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "asset_index": idx,
                    "asset_id": asset_id,
                    "recommendation": (
                        "Leased vehicles belong in Category 13 (downstream "
                        "leased assets), not Category 11 (use of sold products). "
                        "Cat 11 covers products sold, not leased. Report in "
                        "only one category."
                    ),
                })

        logger.info(
            "Double-counting check: %d assets, %d findings",
            len(assets),
            len(findings),
        )

        return findings

    # ==========================================================================
    # Boundary Validation
    # ==========================================================================

    def validate_boundary(self, asset: dict) -> Dict[str, Any]:
        """
        Determine if a leased asset belongs in Category 13, Scope 1/2, or other.

        Classification rules enforce operational control exclusion:
        - Lessor has operational control -> Scope 1/2, NOT Cat 13
        - Lessee has operational control (operating lease) -> Cat 13
        - Finance lease (lessee has control) -> verify control assignment
        - Sub-lease (not original owner) -> Excluded from Cat 13
        - REIT with operational control -> Scope 1/2

        Args:
            asset: Asset dictionary with lease_type, operational_control,
                is_sublease, is_reit, asset_category fields.

        Returns:
            Dictionary with classification, reason, and any warnings.

        Example:
            >>> result = engine.validate_boundary({
            ...     "asset_category": "building",
            ...     "lease_type": "operating_lease",
            ...     "operational_control": False,
            ... })
            >>> result["classification"]
            'CATEGORY_13'
        """
        operational_control = asset.get("operational_control", False)
        lease_type = asset.get("lease_type", "").lower()
        is_sublease = asset.get("is_sublease", False)
        is_reit = asset.get("is_reit", False)
        is_original_owner = asset.get("is_original_owner", True)

        warnings: List[str] = []

        # Rule 1: Lessor has operational control -> Scope 1/2
        if operational_control:
            return {
                "classification": BoundaryClassification.SCOPE_1.value,
                "reason": (
                    "Lessor has operational control over the asset. "
                    "Emissions belong in Scope 1/2, not Category 13."
                ),
                "warnings": warnings,
                "asset": asset,
            }

        # Rule 2: Sub-lease (not original owner) -> Excluded
        if is_sublease and not is_original_owner:
            return {
                "classification": BoundaryClassification.EXCLUDED.value,
                "reason": (
                    "Sub-lease where reporter is not the original asset owner. "
                    "Intermediate lessors do not report under Category 13."
                ),
                "warnings": warnings,
                "asset": asset,
            }

        # Rule 3: REIT with operational control -> Scope 1/2
        if is_reit:
            reit_ops_control = asset.get("reit_operational_control", False)
            if reit_ops_control:
                return {
                    "classification": BoundaryClassification.SCOPE_1.value,
                    "reason": (
                        "REIT property where REIT has operational control. "
                        "Report in Scope 1/2, not Category 13."
                    ),
                    "warnings": warnings,
                    "asset": asset,
                }

        # Rule 4: Finance lease -- verify control
        if lease_type in ("finance_lease", "capital_lease"):
            lessee_has_control = asset.get("lessee_has_operational_control", True)
            if not lessee_has_control:
                warnings.append(
                    "Finance/capital lease with lessor retaining operational "
                    "control. Verify this is correctly classified as Cat 13 "
                    "rather than Scope 1/2."
                )

        # Default: operating lease with lessee control -> Cat 13
        return {
            "classification": BoundaryClassification.CATEGORY_13.value,
            "reason": (
                "Operating lease where lessee has operational control. "
                "Emissions from tenant operations belong in Category 13."
            ),
            "warnings": warnings,
            "asset": asset,
        }

    # ==========================================================================
    # Consolidation Approach Validation
    # ==========================================================================

    def validate_consolidation_approach(
        self, approach: str, assets: list
    ) -> Dict[str, Any]:
        """
        Validate the consolidation approach and its application to assets.

        Checks that:
        - The approach is a recognized GHG Protocol consolidation approach
        - Financial control assets are correctly identified
        - Equity share percentages sum appropriately
        - Operational control boundaries are respected

        Args:
            approach: Consolidation approach identifier.
            assets: List of asset dictionaries with ownership_pct, etc.

        Returns:
            Dictionary with validation result, warnings, and recommendations.

        Example:
            >>> result = engine.validate_consolidation_approach(
            ...     "financial_control",
            ...     [{"asset_id": "A1", "ownership_pct": 100}],
            ... )
            >>> result["valid"]
            True
        """
        valid_approaches = {
            "operational_control", "financial_control", "equity_share",
        }

        approach_lower = approach.lower().strip()
        warnings: List[str] = []
        errors: List[str] = []

        if approach_lower not in valid_approaches:
            return {
                "valid": False,
                "approach": approach,
                "errors": [
                    f"Unrecognized consolidation approach '{approach}'. "
                    f"Must be one of: {', '.join(sorted(valid_approaches))}."
                ],
                "warnings": [],
                "recommendations": [
                    "Use operational_control, financial_control, or equity_share."
                ],
            }

        # Equity share validation: check ownership percentages
        if approach_lower == "equity_share":
            for idx, asset in enumerate(assets):
                ownership_pct = asset.get("ownership_pct")
                if ownership_pct is None:
                    errors.append(
                        f"Asset {idx}: ownership_pct required for equity_share approach."
                    )
                elif not (0 < float(ownership_pct) <= 100):
                    errors.append(
                        f"Asset {idx}: ownership_pct must be between 0 and 100, "
                        f"got {ownership_pct}."
                    )

        # Financial control validation: check control flags
        if approach_lower == "financial_control":
            for idx, asset in enumerate(assets):
                has_financial_control = asset.get("financial_control", None)
                if has_financial_control is None:
                    warnings.append(
                        f"Asset {idx}: financial_control flag not set. "
                        "Assuming financial control exists."
                    )

        # Operational control validation: verify boundary alignment
        if approach_lower == "operational_control":
            for idx, asset in enumerate(assets):
                has_ops_control = asset.get("operational_control", None)
                if has_ops_control is True:
                    warnings.append(
                        f"Asset {idx}: Reporter has operational control. "
                        "Under operational_control approach, this asset's "
                        "emissions should be in Scope 1/2, not Cat 13."
                    )

        is_valid = len(errors) == 0

        recommendations: List[str] = []
        if not is_valid:
            recommendations.append(
                "Fix the identified errors in the consolidation approach."
            )
        if warnings:
            recommendations.append(
                "Review the warnings and verify asset boundary assignments."
            )

        return {
            "valid": is_valid,
            "approach": approach_lower,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    # ==========================================================================
    # Completeness Validation (5-Dimension Scoring)
    # ==========================================================================

    def validate_completeness(self, result: dict) -> Dict[str, Any]:
        """
        Validate completeness of Category 13 emissions reporting.

        Scores across 5 dimensions:
        1. Asset coverage: % of leased assets with emission data
        2. Temporal coverage: % of reporting period with data
        3. Geographic coverage: % of regions/countries covered
        4. Category coverage: % of asset categories reported
        5. Method coverage: appropriate methods for each asset type

        Args:
            result: Calculation result dictionary containing coverage metrics.

        Returns:
            Dictionary with dimension scores, overall score, and recommendations.

        Example:
            >>> completeness = engine.validate_completeness(result)
            >>> completeness["overall_score"]
            85.0
        """
        scores: Dict[str, Decimal] = {}
        details: Dict[str, str] = {}

        # Dimension 1: Asset coverage
        total_assets = result.get("total_leased_assets", 0)
        covered_assets = result.get("assets_with_data", 0)
        if total_assets > 0:
            asset_coverage = (
                Decimal(str(covered_assets)) / Decimal(str(total_assets)) * Decimal("100")
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
            scores["asset_coverage"] = asset_coverage
            details["asset_coverage"] = (
                f"{covered_assets}/{total_assets} assets with emission data"
            )
        else:
            scores["asset_coverage"] = Decimal("0")
            details["asset_coverage"] = "No leased assets reported"

        # Dimension 2: Temporal coverage
        reporting_months = result.get("reporting_months_covered", 0)
        total_months = result.get("total_reporting_months", 12)
        if total_months > 0:
            temporal_coverage = (
                Decimal(str(reporting_months)) / Decimal(str(total_months)) * Decimal("100")
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
            scores["temporal_coverage"] = temporal_coverage
            details["temporal_coverage"] = (
                f"{reporting_months}/{total_months} months covered"
            )
        else:
            scores["temporal_coverage"] = Decimal("0")
            details["temporal_coverage"] = "No temporal data"

        # Dimension 3: Geographic coverage
        total_regions = result.get("total_regions", 0)
        covered_regions = result.get("regions_with_data", 0)
        if total_regions > 0:
            geo_coverage = (
                Decimal(str(covered_regions)) / Decimal(str(total_regions)) * Decimal("100")
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
            scores["geographic_coverage"] = geo_coverage
            details["geographic_coverage"] = (
                f"{covered_regions}/{total_regions} regions covered"
            )
        else:
            scores["geographic_coverage"] = Decimal("100")
            details["geographic_coverage"] = "Single region or not applicable"

        # Dimension 4: Category coverage
        all_categories = {"building", "vehicle", "equipment", "it_asset"}
        reported_categories = set(result.get("reported_categories", []))
        relevant_categories = set(result.get("relevant_categories", list(all_categories)))
        if relevant_categories:
            cat_covered = len(reported_categories & relevant_categories)
            cat_total = len(relevant_categories)
            cat_coverage = (
                Decimal(str(cat_covered)) / Decimal(str(cat_total)) * Decimal("100")
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
            scores["category_coverage"] = cat_coverage
            details["category_coverage"] = (
                f"{cat_covered}/{cat_total} asset categories reported"
            )
        else:
            scores["category_coverage"] = Decimal("100")
            details["category_coverage"] = "No relevant categories defined"

        # Dimension 5: Method coverage
        method_appropriate = result.get("method_coverage_score")
        if method_appropriate is not None:
            scores["method_coverage"] = Decimal(str(method_appropriate)).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            details["method_coverage"] = (
                f"Method appropriateness score: {method_appropriate}%"
            )
        else:
            scores["method_coverage"] = Decimal("50")
            details["method_coverage"] = "Method coverage not assessed (default 50%)"

        # Overall score (equal weight for all 5 dimensions)
        if scores:
            overall = (
                sum(scores.values()) / Decimal(str(len(scores)))
            ).quantize(_QUANT_2DP, rounding=ROUNDING)
        else:
            overall = Decimal("0")

        # Recommendations based on weak dimensions
        recommendations: List[str] = []
        for dim, score in scores.items():
            if score < Decimal("50"):
                dim_label = dim.replace("_", " ").title()
                recommendations.append(
                    f"CRITICAL: {dim_label} is only {score}%. "
                    f"Immediate improvement required."
                )
            elif score < Decimal("80"):
                dim_label = dim.replace("_", " ").title()
                recommendations.append(
                    f"IMPROVEMENT: {dim_label} is {score}%. "
                    f"Target 80%+ for robust reporting."
                )

        return {
            "overall_score": float(overall),
            "dimension_scores": {k: float(v) for k, v in scores.items()},
            "dimension_details": details,
            "recommendations": recommendations,
            "status": (
                "PASS" if overall >= Decimal("80")
                else "WARNING" if overall >= Decimal("50")
                else "FAIL"
            ),
        }

    # ==========================================================================
    # Tenant Data Coverage Validation
    # ==========================================================================

    def validate_tenant_data_coverage(
        self, assets: list
    ) -> Dict[str, Any]:
        """
        Validate the percentage of leased assets with actual tenant metered data.

        Assesses data quality by categorizing assets into:
        - Metered: actual tenant energy data available
        - Estimated: benchmark/proxy data used
        - Missing: no data available

        Args:
            assets: List of asset dictionaries with data_source field
                (metered, estimated, benchmark, missing, none).

        Returns:
            Dictionary with coverage percentages and recommendations.

        Example:
            >>> coverage = engine.validate_tenant_data_coverage(assets)
            >>> coverage["metered_pct"]
            65.0
        """
        total = len(assets)
        if total == 0:
            return {
                "total_assets": 0,
                "metered_count": 0,
                "estimated_count": 0,
                "missing_count": 0,
                "metered_pct": 0.0,
                "estimated_pct": 0.0,
                "missing_pct": 0.0,
                "overall_coverage_pct": 0.0,
                "data_quality_tier": "NONE",
                "recommendations": ["No assets provided for coverage assessment."],
            }

        metered_count = 0
        estimated_count = 0
        missing_count = 0

        for asset in assets:
            data_source = str(asset.get("data_source", "missing")).lower()
            if data_source in ("metered", "actual", "measured"):
                metered_count += 1
            elif data_source in ("estimated", "benchmark", "proxy", "modeled"):
                estimated_count += 1
            else:
                missing_count += 1

        metered_pct = float(
            (Decimal(str(metered_count)) / Decimal(str(total)) * Decimal("100"))
            .quantize(_QUANT_2DP, rounding=ROUNDING)
        )
        estimated_pct = float(
            (Decimal(str(estimated_count)) / Decimal(str(total)) * Decimal("100"))
            .quantize(_QUANT_2DP, rounding=ROUNDING)
        )
        missing_pct = float(
            (Decimal(str(missing_count)) / Decimal(str(total)) * Decimal("100"))
            .quantize(_QUANT_2DP, rounding=ROUNDING)
        )

        overall_coverage = metered_pct + estimated_pct

        # Determine data quality tier
        if metered_pct >= 80:
            data_quality_tier = "HIGH"
        elif metered_pct >= 50:
            data_quality_tier = "MEDIUM"
        elif overall_coverage >= 50:
            data_quality_tier = "LOW"
        else:
            data_quality_tier = "VERY_LOW"

        recommendations: List[str] = []
        if metered_pct < 50:
            recommendations.append(
                "Less than 50% of assets have metered data. "
                "Implement smart metering and green lease data-sharing clauses."
            )
        if missing_pct > 20:
            recommendations.append(
                f"{missing_pct:.1f}% of assets have no data. "
                "Use benchmarks (EUI by building type) as interim measures."
            )
        if data_quality_tier in ("LOW", "VERY_LOW"):
            recommendations.append(
                "Low data quality tier. Consider tenant engagement programs "
                "and automated data collection from utility providers."
            )

        return {
            "total_assets": total,
            "metered_count": metered_count,
            "estimated_count": estimated_count,
            "missing_count": missing_count,
            "metered_pct": metered_pct,
            "estimated_pct": estimated_pct,
            "missing_pct": missing_pct,
            "overall_coverage_pct": overall_coverage,
            "data_quality_tier": data_quality_tier,
            "recommendations": recommendations,
        }

    # ==========================================================================
    # Report Generation
    # ==========================================================================

    def generate_report(
        self, results: Dict[str, ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive compliance report with weighted overall score.

        Aggregates framework results, calculates the weighted overall compliance
        score, categorizes findings by severity, and produces actionable
        recommendations.

        Args:
            results: Dictionary mapping framework name to ComplianceCheckResult.

        Returns:
            Report dictionary with overall_score, overall_status,
            framework_scores, findings by severity, and recommendations.

        Example:
            >>> report = engine.generate_report(all_results)
            >>> report["overall_score"]
            85.5
            >>> report["overall_status"]
            'warning'
        """
        return self.get_compliance_summary(results)

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

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _has_building_assets(self, result: dict) -> bool:
        """
        Check if the result contains building-type leased assets.

        Args:
            result: Calculation result dictionary.

        Returns:
            True if building assets are present, False otherwise.
        """
        asset_breakdown = (
            result.get("asset_breakdown")
            or result.get("by_asset_category")
            or {}
        )
        if isinstance(asset_breakdown, dict):
            return "building" in asset_breakdown or "buildings" in asset_breakdown

        asset_categories = result.get("asset_categories", [])
        if isinstance(asset_categories, list):
            return "building" in [c.lower() for c in asset_categories]

        return False

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
            'compliance_checker_engine'
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "check_count": self._check_count,
            "enabled_frameworks": self._enabled_frameworks,
            "strict_mode": self._strict_mode,
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
    "AGENT_ID",
    "AGENT_COMPONENT",
    "ComplianceSeverity",
    "ComplianceStatus",
    "ComplianceFramework",
    "DoubleCountingCategory",
    "BoundaryClassification",
    "ConsolidationApproach",
    "LeaseType",
    "AssetCategory",
    "AllocationMethod",
    "ComplianceFinding",
    "ComplianceCheckResult",
    "FrameworkCheckState",
    "FRAMEWORK_WEIGHTS",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "BUILDING_EUI_BENCHMARKS",
    "EPC_RATING_SCORES",
    "ComplianceCheckerEngine",
    "get_compliance_checker",
]
