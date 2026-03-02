# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-020 Engine 6

This module implements regulatory compliance checking for Employee Commuting
emissions (GHG Protocol Scope 3 Category 7) against 7 regulatory frameworks.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 7 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 Climate Change (telework disclosure required)
4. CDP Climate Change Questionnaire (C6.5 Employee Commuting, mode share reporting)
5. SBTi (Science Based Targets initiative, materiality >1% of Scope 3)
6. SB 253 (California Climate Corporate Data Accountability Act)
7. GRI 305 Emissions Standard

Category 7-Specific Compliance Rules:
- Telework disclosure requirements (CSRD specifically asks for this)
- Mode share reporting validation (CDP requires mode breakdown)
- Survey methodology documentation check
- Materiality threshold validation (SBTi: Category 7 > 1% of Scope 3)
- Base year recalculation policy check
- Third-party assurance readiness (SB 253)
- Active transport zero-emissions enforcement
- WTT boundary alignment with Category 3

Double-Counting Prevention Rules (10 rules: DC-EC-001 through DC-EC-010):
    DC-EC-001: Exclude company-owned/leased vehicles (should be Scope 1)
    DC-EC-002: Exclude business travel trips (should be Category 6)
    DC-EC-003: Exclude company shuttle services (Scope 1 or Cat 4)
    DC-EC-004: Telework energy not double-counted with Scope 2
    DC-EC-005: No overlap with Category 6 business travel
    DC-EC-006: Avoid commute counted as Category 4 freight
    DC-EC-007: Active transport zero emissions (not negative)
    DC-EC-008: WTT not double-counted with Category 3
    DC-EC-009: Shuttle to transit hub allocation
    DC-EC-010: EV charging at office not in Category 7

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-007
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.employee_commuting.models import (
    ComplianceFramework,
    ComplianceStatus,
    ComplianceCheckResult,
    CommuteMode,
    VehicleType,
    FRAMEWORK_REQUIRED_DISCLOSURES,
    calculate_provenance_hash,
)
from greenlang.employee_commuting.config import get_config
from greenlang.employee_commuting.metrics import get_metrics

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# Active transport modes that MUST have zero emissions
ACTIVE_TRANSPORT_MODES: Set[str] = {
    CommuteMode.CYCLING.value,
    CommuteMode.WALKING.value,
}

# Modes that involve personal vehicles (Scope 1 boundary risk)
PERSONAL_VEHICLE_MODES: Set[str] = {
    CommuteMode.SOV.value,
    CommuteMode.CARPOOL.value,
    CommuteMode.MOTORCYCLE.value,
}

# Modes that represent company-provided transport
COMPANY_TRANSPORT_MODES: Set[str] = {
    "company_shuttle",
    "company_bus",
    "shuttle",
}

# Modes that involve electric charging
ELECTRIC_MODES: Set[str] = {
    CommuteMode.E_BIKE.value,
    CommuteMode.E_SCOOTER.value,
    "bev",
    "plugin_hybrid",
}


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
    """Scope 3 categories that could overlap with Category 7."""

    SCOPE_1 = "SCOPE_1"  # Company-owned/leased vehicles
    SCOPE_2 = "SCOPE_2"  # Office EV charging / telework grid
    CATEGORY_3 = "CATEGORY_3"  # Fuel & energy activities (WTT)
    CATEGORY_4 = "CATEGORY_4"  # Upstream transportation (shuttles)
    CATEGORY_6 = "CATEGORY_6"  # Business travel
    CATEGORY_9 = "CATEGORY_9"  # Downstream transportation


class BoundaryClassification(str, Enum):
    """Category boundary classification for a commute trip."""

    CATEGORY_7 = "CATEGORY_7"  # Employee commuting (correct)
    CATEGORY_6 = "CATEGORY_6"  # Business travel
    SCOPE_1 = "SCOPE_1"  # Company-owned vehicle
    SCOPE_2 = "SCOPE_2"  # Office EV charging
    CATEGORY_4 = "CATEGORY_4"  # Upstream transportation (shuttle)
    CATEGORY_3 = "CATEGORY_3"  # Fuel & energy activities (WTT)
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
    Compliance checker for Employee Commuting emissions (Category 7).

    Validates calculation results against 7 regulatory frameworks with
    Category 7-specific rules including telework disclosure, mode share
    reporting, survey methodology documentation, double-counting prevention
    (10 rules), and category boundary enforcement.

    Thread Safety:
        Singleton pattern with threading.Lock for concurrent access.

    Attributes:
        _config: Employee commuting configuration (compliance section)
        _metrics: Prometheus metrics instance for compliance check tracking
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

    def __new__(cls) -> "ComplianceCheckerEngine":
        """Create or return singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine with configuration."""
        if self._initialized:
            return
        self._initialized = True

        self._config = get_config()
        self._metrics = get_metrics()
        self._enabled_frameworks: List[str] = (
            self._config.compliance.get_frameworks()
        )
        self._check_count: int = 0

        logger.info(
            "ComplianceCheckerEngine initialized: version=%s, "
            "frameworks=%d, strict_mode=%s, telework_disclosure=%s, "
            "mode_share_required=%s, double_counting_check=%s",
            ENGINE_VERSION,
            len(self._enabled_frameworks),
            self._config.compliance.strict_mode,
            self._config.compliance.telework_disclosure_required,
            self._config.compliance.mode_share_required,
            self._config.compliance.double_counting_check,
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
        self,
        result: dict,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Run all enabled framework checks and return results.

        Iterates over each enabled framework and dispatches to the
        appropriate check method. Errors in one framework do not
        prevent other frameworks from being checked. An optional
        frameworks list can be passed to restrict which frameworks
        are checked.

        Args:
            result: Calculation result dictionary containing total_co2e,
                mode_breakdown, telework_emissions, method, ef_sources, etc.
            frameworks: Optional list of framework names to check. If None,
                all enabled frameworks from config are checked.

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
            "GRI_305": (ComplianceFramework.GRI, "check_gri_305"),
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
            "gri": (ComplianceFramework.GRI, "check_gri_305"),
            # Map EPA_CCL to GRI as a placeholder
            "EPA_CCL": (ComplianceFramework.GRI, "check_gri_305"),
        }

        target_frameworks = frameworks if frameworks else self._enabled_frameworks

        for framework_name in target_frameworks:
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
                            "description": (
                                f"Compliance check failed: {str(e)}"
                            ),
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

        # Record metrics
        try:
            self._metrics.record_calculation(
                mode="compliance",
                method="all_frameworks",
                status="success",
                duration=duration,
                co2e=0.0,
                tenant_id="system",
            )
        except Exception:
            pass  # Metrics failures must not affect compliance logic

        logger.info(
            "All framework checks complete: %d frameworks, duration=%.4fs",
            len(all_results),
            duration,
        )

        return all_results

    # ==========================================================================
    # Framework: GHG Protocol Scope 3 (Category 7)
    # ==========================================================================

    def check_ghg_protocol(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard (Category 7).

        Category 7 specific checks:
            - GHG-EC-001: Total CO2e present and > 0
            - GHG-EC-002: Calculation method documented
            - GHG-EC-003: Emission factor sources documented
            - GHG-EC-004: Exclusions documented
            - GHG-EC-005: DQI score present
            - GHG-EC-006: Mode breakdown present
            - GHG-EC-007: Survey methodology documented (if survey-based)
            - GHG-EC-008: Telework emissions separately reported
            - GHG-EC-009: Employee count documented
            - GHG-EC-010: Working days documented

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

        # GHG-EC-001: Total emissions present and positive
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GHG-EC-001", "Total CO2e is present and positive")
        else:
            state.add_fail(
                "GHG-EC-001",
                "Total CO2e is missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Ensure total_co2e is calculated and > 0. "
                    "Category 7 must report total employee commuting emissions."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7",
            )

        # GHG-EC-002: Calculation method documented
        method = result.get("method") or result.get("calculation_method")
        if method:
            valid_methods = {
                "employee_specific", "average_data", "spend_based",
                "distance_based", "fuel_based", "survey_based",
            }
            method_lower = str(method).lower().replace("-", "_").replace(" ", "_")
            if method_lower in valid_methods:
                state.add_pass(
                    "GHG-EC-002",
                    f"Calculation method documented: {method}",
                )
            else:
                state.add_warning(
                    "GHG-EC-002",
                    f"Calculation method '{method}' is non-standard",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Use one of the GHG Protocol recommended methods: "
                        "employee-specific (survey), average-data, or spend-based."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Table 7.1",
                )
        else:
            state.add_fail(
                "GHG-EC-002",
                "Calculation method not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation method used: "
                    "employee-specific (survey), average-data, or spend-based."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-EC-003: Emission factor sources documented
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GHG-EC-003", "Emission factor sources documented")
        else:
            state.add_fail(
                "GHG-EC-003",
                "Emission factor sources not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document all emission factor sources "
                    "(DEFRA, EPA, IEA, Census, EEIO, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7",
            )

        # GHG-EC-004: Exclusions documented
        exclusions = result.get("exclusions")
        if exclusions is not None:
            state.add_pass("GHG-EC-004", "Exclusions documented")
        else:
            state.add_warning(
                "GHG-EC-004",
                "Exclusions not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document any exclusions from Category 7 reporting "
                    "(e.g., contractors, part-time employees below threshold, "
                    "employees at sites < 50 FTEs)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7",
            )

        # GHG-EC-005: Data quality indicator score
        dqi_score = result.get("dqi_score") or result.get("data_quality_score")
        if dqi_score is not None:
            state.add_pass("GHG-EC-005", "DQI score present")
        else:
            state.add_warning(
                "GHG-EC-005",
                "Data quality indicator (DQI) score not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate and report DQI scores across 5 dimensions "
                    "(representativeness, completeness, temporal, geographical, "
                    "technological)."
                ),
                regulation_reference="GHG Protocol Scope 3, Table 7.1",
            )

        # GHG-EC-006: Mode breakdown present
        mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
        if (
            mode_breakdown
            and isinstance(mode_breakdown, dict)
            and len(mode_breakdown) > 0
        ):
            state.add_pass("GHG-EC-006", "Mode breakdown present")
        else:
            state.add_fail(
                "GHG-EC-006",
                "Commute mode breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by commute mode "
                    "(SOV, carpool, bus, metro, rail, cycling, walking, "
                    "telework, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7",
            )

        # GHG-EC-007: Survey methodology documented (if survey-based)
        method_lower = str(method).lower() if method else ""
        is_survey = "survey" in method_lower or "employee_specific" in method_lower
        if is_survey:
            survey_method = (
                result.get("survey_methodology")
                or result.get("survey_method")
                or result.get("survey_type")
            )
            survey_response_rate = result.get("survey_response_rate")
            if survey_method:
                state.add_pass(
                    "GHG-EC-007",
                    "Survey methodology documented",
                )
            else:
                state.add_fail(
                    "GHG-EC-007",
                    "Survey methodology not documented (required for survey-based method)",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Document the survey methodology: full census, "
                        "stratified sample, random sample, or convenience. "
                        "Include sample size and response rate."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 7",
                )
            if survey_response_rate is not None:
                state.add_pass(
                    "GHG-EC-007b",
                    f"Survey response rate documented: {survey_response_rate}",
                )
            else:
                state.add_warning(
                    "GHG-EC-007b",
                    "Survey response rate not documented",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Document the survey response rate. Higher response "
                        "rates improve data quality and reduce uncertainty."
                    ),
                    regulation_reference="GHG Protocol Scope 3, Ch 7",
                )

        # GHG-EC-008: Telework emissions separately reported
        telework_emissions = (
            result.get("telework_emissions")
            or result.get("telework_co2e")
            or result.get("remote_work_emissions")
        )
        telework_employees = (
            result.get("telework_employees")
            or result.get("remote_workers")
        )
        if telework_emissions is not None:
            state.add_pass(
                "GHG-EC-008",
                "Telework emissions separately reported",
            )
        else:
            state.add_warning(
                "GHG-EC-008",
                "Telework emissions not separately reported",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Report telework/remote work emissions separately. "
                    "GHG Protocol recommends disclosing home-office energy "
                    "consumption for remote workers."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7 Guidance",
            )

        # GHG-EC-009: Employee count documented
        employee_count = (
            result.get("employee_count")
            or result.get("total_employees")
            or result.get("fte_count")
        )
        if employee_count is not None:
            state.add_pass("GHG-EC-009", "Employee count documented")
        else:
            state.add_warning(
                "GHG-EC-009",
                "Employee count not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document the total number of employees included in the "
                    "Category 7 calculation for intensity metrics."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7",
            )

        # GHG-EC-010: Working days documented
        working_days = (
            result.get("working_days_per_year")
            or result.get("working_days")
            or result.get("commute_days")
        )
        if working_days is not None:
            state.add_pass("GHG-EC-010", "Working days documented")
        else:
            state.add_warning(
                "GHG-EC-010",
                "Working days per year not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document the number of working days assumed per year. "
                    "Regional defaults vary (US: 225, UK: 212, DE: 200)."
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 7",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: ISO 14064
    # ==========================================================================

    def check_iso_14064(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018 (Clause 5.2.4).

        Checks:
            - ISO-EC-001: Total CO2e present
            - ISO-EC-002: Uncertainty analysis present
            - ISO-EC-003: Base year documented
            - ISO-EC-004: Methodology described
            - ISO-EC-005: Reporting period defined
            - ISO-EC-006: Organizational boundary documented
            - ISO-EC-007: Quantification approach documented

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

        # ISO-EC-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("ISO-EC-001", "Total CO2e present")
        else:
            state.add_fail(
                "ISO-EC-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Calculate and report total CO2e emissions for "
                    "employee commuting (Category 7)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # ISO-EC-002: Uncertainty analysis
        uncertainty = (
            result.get("uncertainty_analysis")
            or result.get("uncertainty")
            or result.get("uncertainty_percentage")
        )
        if uncertainty is not None:
            state.add_pass("ISO-EC-002", "Uncertainty analysis present")
        else:
            state.add_fail(
                "ISO-EC-002",
                "Uncertainty analysis not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Perform and document uncertainty analysis "
                    "(Monte Carlo, analytical, or IPCC Tier 2). "
                    "Employee commuting data often has high uncertainty "
                    "due to survey-based collection."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 9",
            )

        # ISO-EC-003: Base year documented
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("ISO-EC-003", "Base year documented")
        else:
            state.add_fail(
                "ISO-EC-003",
                "Base year not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the base year for emissions comparison "
                    "and trend analysis. Include base year recalculation "
                    "policy for structural changes (e.g., office closures, "
                    "remote work policy changes)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # ISO-EC-004: Methodology described
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("ISO-EC-004", "Methodology described")
        else:
            state.add_fail(
                "ISO-EC-004",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the quantification methodology including "
                    "emission factors, data sources, calculation approach, "
                    "and survey design (if applicable)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2",
            )

        # ISO-EC-005: Reporting period defined
        reporting_period = (
            result.get("reporting_period")
            or result.get("period")
        )
        if reporting_period:
            state.add_pass("ISO-EC-005", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-EC-005",
                "Reporting period not specified",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Specify the reporting period (e.g., 2025, 2025-Q3)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-EC-006: Organizational boundary documented
        org_boundary = (
            result.get("organizational_boundary")
            or result.get("org_boundary")
            or result.get("boundary_approach")
        )
        if org_boundary:
            state.add_pass("ISO-EC-006", "Organizational boundary documented")
        else:
            state.add_warning(
                "ISO-EC-006",
                "Organizational boundary not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the organizational boundary approach "
                    "(equity share, financial control, or operational control) "
                    "and which employee populations are included."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # ISO-EC-007: Quantification approach
        quant_approach = (
            result.get("quantification_approach")
            or result.get("data_sources")
        )
        if quant_approach:
            state.add_pass("ISO-EC-007", "Quantification approach documented")
        else:
            state.add_warning(
                "ISO-EC-007",
                "Quantification approach not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document whether primary data (surveys) or secondary "
                    "data (regional averages) were used and the rationale."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 6",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CSRD / ESRS E1
    # ==========================================================================

    def check_csrd_esrs(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CSRD ESRS E1 Climate Change.

        CSRD/ESRS has specific requirements for employee commuting including
        mandatory telework disclosure and mode share reporting.

        Checks:
            - CSRD-EC-001: Total CO2e by category
            - CSRD-EC-002: Methodology description
            - CSRD-EC-003: Targets documented
            - CSRD-EC-004: Mode breakdown present
            - CSRD-EC-005: Reduction actions described
            - CSRD-EC-006: Telework policy and emissions disclosed (CSRD-specific)
            - CSRD-EC-007: Intensity metrics present
            - CSRD-EC-008: Year-over-year comparison

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

        # CSRD-EC-001: Total emissions by category
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CSRD-EC-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CSRD-EC-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Scope 3 Category 7 emissions. "
                    "CSRD requires disclosure of all material Scope 3 categories."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )

        # CSRD-EC-002: Methodology description
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("CSRD-EC-002", "Methodology described")
        else:
            state.add_fail(
                "CSRD-EC-002",
                "Methodology not described",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Describe the calculation methodology for employee "
                    "commuting emissions as required by ESRS E1. Include "
                    "data collection approach (survey vs average-data)."
                ),
                regulation_reference="ESRS E1-6, para 53",
            )

        # CSRD-EC-003: Targets documented
        targets = result.get("targets") or result.get("reduction_targets")
        if targets:
            state.add_pass("CSRD-EC-003", "Targets documented")
        else:
            state.add_warning(
                "CSRD-EC-003",
                "Reduction targets not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission reduction targets for employee commuting "
                    "(e.g., increase transit mode share, promote telework, "
                    "incentivize EV adoption, expand cycling infrastructure)."
                ),
                regulation_reference="ESRS E1-4, para 34",
            )

        # CSRD-EC-004: Mode breakdown present
        mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
        if (
            mode_breakdown
            and isinstance(mode_breakdown, dict)
            and len(mode_breakdown) > 0
        ):
            state.add_pass("CSRD-EC-004", "Mode breakdown present")
        else:
            state.add_fail(
                "CSRD-EC-004",
                "Commute mode breakdown not provided",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Provide emissions breakdown by commute mode "
                    "(car, transit, cycling, walking, telework, etc.)."
                ),
                regulation_reference="ESRS E1-6, para 51(d)",
            )

        # CSRD-EC-005: Actions described
        actions = result.get("actions") or result.get("reduction_actions")
        if actions:
            state.add_pass("CSRD-EC-005", "Reduction actions described")
        else:
            state.add_warning(
                "CSRD-EC-005",
                "Reduction actions not described",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Describe actions taken or planned to reduce employee "
                    "commuting emissions (e.g., remote work policy, cycling "
                    "facilities, transit subsidies, EV charging at office, "
                    "carpooling programs)."
                ),
                regulation_reference="ESRS E1-3, para 29",
            )

        # CSRD-EC-006: Telework policy and emissions (CSRD-SPECIFIC)
        telework_policy = (
            result.get("telework_policy")
            or result.get("remote_work_policy")
        )
        telework_emissions = (
            result.get("telework_emissions")
            or result.get("telework_co2e")
            or result.get("remote_work_emissions")
        )
        telework_percentage = (
            result.get("telework_percentage")
            or result.get("remote_work_percentage")
            or result.get("telework_rate")
        )

        if telework_policy and telework_emissions is not None:
            state.add_pass(
                "CSRD-EC-006",
                "Telework policy and emissions disclosed",
            )
        elif telework_emissions is not None:
            state.add_warning(
                "CSRD-EC-006",
                "Telework emissions present but policy not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the company telework/remote work policy. "
                    "CSRD/ESRS specifically requires disclosure of telework "
                    "arrangements and their emissions impact."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )
        elif telework_policy:
            state.add_warning(
                "CSRD-EC-006",
                "Telework policy documented but emissions not quantified",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Quantify telework emissions (home-office energy use). "
                    "CSRD/ESRS requires disclosure of remote work emissions."
                ),
                regulation_reference="ESRS E1-6, para 51",
            )
        else:
            # Check if config requires telework disclosure
            if self._config.compliance.telework_disclosure_required:
                state.add_fail(
                    "CSRD-EC-006",
                    "Telework disclosure missing (required by CSRD)",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Disclose telework/remote work policy AND quantify "
                        "home-office energy emissions. CSRD/ESRS E1 "
                        "specifically requires telework disclosure."
                    ),
                    regulation_reference="ESRS E1-6, para 51",
                )
            else:
                state.add_warning(
                    "CSRD-EC-006",
                    "Telework disclosure not provided",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Consider disclosing telework/remote work policy "
                        "and emissions for CSRD compliance."
                    ),
                    regulation_reference="ESRS E1-6, para 51",
                )

        # CSRD-EC-007: Intensity metrics
        intensity = (
            result.get("intensity_per_employee")
            or result.get("co2e_per_fte")
            or result.get("emissions_intensity")
        )
        if intensity is not None:
            state.add_pass("CSRD-EC-007", "Intensity metrics present")
        else:
            state.add_warning(
                "CSRD-EC-007",
                "Emissions intensity per employee not provided",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Calculate and report emissions intensity "
                    "(tCO2e per employee or per FTE) for benchmarking."
                ),
                regulation_reference="ESRS E1-6, para 54",
            )

        # CSRD-EC-008: Year-over-year comparison
        yoy_change = (
            result.get("year_over_year_change")
            or result.get("yoy_change")
            or result.get("trend")
        )
        if yoy_change is not None:
            state.add_pass("CSRD-EC-008", "Year-over-year comparison present")
        else:
            state.add_warning(
                "CSRD-EC-008",
                "Year-over-year comparison not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Include year-over-year comparison of employee commuting "
                    "emissions to demonstrate trends and progress."
                ),
                regulation_reference="ESRS E1-6, para 52",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: CDP Climate Change
    # ==========================================================================

    def check_cdp(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change Questionnaire (C6.5).

        CDP has specific requirements for Category 7 including mode share
        reporting and employee commuting-specific disclosure.

        Checks:
            - CDP-EC-001: Total CO2e present
            - CDP-EC-002: Mode share breakdown (CDP C6.5 requires this)
            - CDP-EC-003: Employee count for intensity
            - CDP-EC-004: Verification status
            - CDP-EC-005: Survey coverage documented
            - CDP-EC-006: Telework percentage reported

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

        # CDP-EC-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("CDP-EC-001", "Total CO2e reported")
        else:
            state.add_fail(
                "CDP-EC-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 7 emissions to CDP. "
                    "Employee commuting is a standard CDP C6.5 disclosure."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-EC-002: Mode share breakdown (CDP CRITICAL)
        mode_breakdown = result.get("mode_breakdown") or result.get("by_mode")
        mode_shares = result.get("mode_shares") or result.get("mode_split")
        if (
            mode_breakdown
            and isinstance(mode_breakdown, dict)
            and len(mode_breakdown) > 0
        ):
            # Validate mode shares sum to ~100% if provided
            if mode_shares and isinstance(mode_shares, dict):
                total_share = sum(
                    Decimal(str(v)) for v in mode_shares.values()
                )
                if (
                    total_share >= Decimal("0.95")
                    and total_share <= Decimal("1.05")
                ):
                    state.add_pass(
                        "CDP-EC-002",
                        "Mode share breakdown present and valid",
                    )
                else:
                    state.add_warning(
                        "CDP-EC-002",
                        f"Mode shares do not sum to ~100% (sum={total_share})",
                        ComplianceSeverity.MEDIUM,
                        recommendation=(
                            "Ensure mode share percentages sum to approximately "
                            "100%. CDP expects a complete breakdown."
                        ),
                        regulation_reference="CDP CC Module C6.5",
                    )
            else:
                state.add_pass(
                    "CDP-EC-002",
                    "Mode breakdown present",
                )
        else:
            # CDP requires mode share reporting for Category 7
            if self._config.compliance.mode_share_required:
                state.add_fail(
                    "CDP-EC-002",
                    "Mode share breakdown not provided (required by CDP C6.5)",
                    ComplianceSeverity.HIGH,
                    recommendation=(
                        "Provide commute mode share breakdown "
                        "(percentage by SOV, carpool, transit, cycling, "
                        "walking, telework). CDP C6.5 requires mode reporting."
                    ),
                    regulation_reference="CDP CC Module C6.5",
                )
            else:
                state.add_warning(
                    "CDP-EC-002",
                    "Mode share breakdown not provided",
                    ComplianceSeverity.MEDIUM,
                    recommendation=(
                        "Provide commute mode share breakdown for CDP."
                    ),
                    regulation_reference="CDP CC Module C6.5",
                )

        # CDP-EC-003: Employee count for intensity
        employee_count = (
            result.get("employee_count")
            or result.get("total_employees")
            or result.get("fte_count")
        )
        if employee_count is not None:
            state.add_pass("CDP-EC-003", "Employee count documented")
        else:
            state.add_warning(
                "CDP-EC-003",
                "Employee count not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Report total employee count for per-capita emissions "
                    "intensity. CDP uses this for benchmarking."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-EC-004: Verification status
        verification = (
            result.get("verification_status")
            or result.get("verified")
            or result.get("assurance")
        )
        if verification is not None:
            state.add_pass("CDP-EC-004", "Verification status documented")
        else:
            state.add_warning(
                "CDP-EC-004",
                "Verification status not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document whether employee commuting emissions have been "
                    "third-party verified. CDP scores verification status."
                ),
                regulation_reference="CDP CC Module C10.1",
            )

        # CDP-EC-005: Survey coverage
        survey_coverage = (
            result.get("survey_coverage")
            or result.get("survey_response_rate")
            or result.get("data_coverage")
        )
        if survey_coverage is not None:
            state.add_pass("CDP-EC-005", "Survey coverage documented")
        else:
            state.add_warning(
                "CDP-EC-005",
                "Survey coverage not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document the percentage of employees covered by "
                    "the commuting survey or data collection method."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        # CDP-EC-006: Telework percentage
        telework_pct = (
            result.get("telework_percentage")
            or result.get("remote_work_percentage")
            or result.get("telework_rate")
        )
        if telework_pct is not None:
            state.add_pass("CDP-EC-006", "Telework percentage reported")
        else:
            state.add_warning(
                "CDP-EC-006",
                "Telework/remote work percentage not reported",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Report the percentage of employees who telework "
                    "and the associated emissions impact."
                ),
                regulation_reference="CDP CC Module C6.5",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SBTi
    # ==========================================================================

    def check_sbti(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative.

        SBTi requires Category 7 to be included in target boundary
        if it exceeds 1% of total Scope 3 emissions.

        Checks:
            - SBTI-EC-001: Total CO2e present
            - SBTI-EC-002: Target coverage documented
            - SBTI-EC-003: Progress tracking present
            - SBTI-EC-004: Materiality assessment (> 1% of Scope 3)
            - SBTI-EC-005: Reduction initiatives documented
            - SBTI-EC-006: Base year recalculation policy

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

        # SBTI-EC-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SBTI-EC-001", "Total CO2e present for SBTi")
        else:
            state.add_fail(
                "SBTI-EC-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Calculate total Category 7 emissions for SBTi "
                    "target boundary assessment."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-EC-002: Target coverage
        target_coverage = (
            result.get("target_coverage")
            or result.get("sbti_coverage")
        )
        if target_coverage is not None:
            state.add_pass("SBTI-EC-002", "Target coverage documented")
        else:
            state.add_warning(
                "SBTI-EC-002",
                "SBTi target coverage not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document what percentage of Scope 3 emissions is "
                    "covered by SBTi targets (minimum 67% required). "
                    "If Category 7 is material, it must be in the boundary."
                ),
                regulation_reference="SBTi Criteria v5.1, C20",
            )

        # SBTI-EC-003: Progress tracking
        progress = (
            result.get("progress_tracking")
            or result.get("year_over_year_change")
            or result.get("trend")
        )
        if progress is not None:
            state.add_pass("SBTI-EC-003", "Progress tracking present")
        else:
            state.add_warning(
                "SBTI-EC-003",
                "Progress tracking not present",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Track year-over-year emissions change for Category 7 "
                    "to demonstrate progress toward SBTi targets."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-EC-004: Materiality assessment (> 1% of Scope 3)
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat7_decimal = Decimal(str(total_co2e))
                scope3_decimal = Decimal(str(total_scope3))
                if scope3_decimal > 0:
                    cat7_pct = (
                        cat7_decimal / scope3_decimal * Decimal("100")
                    )
                    if cat7_pct >= Decimal("1"):
                        state.add_pass(
                            "SBTI-EC-004",
                            (
                                f"Category 7 is material "
                                f"({cat7_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                        )
                    else:
                        state.add_warning(
                            "SBTI-EC-004",
                            (
                                f"Category 7 below materiality threshold "
                                f"({cat7_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                            ComplianceSeverity.LOW,
                            recommendation=(
                                "Category 7 may not need to be included "
                                "in SBTi target boundary if below 1%. "
                                "However, continued monitoring is recommended."
                            ),
                        )
                else:
                    state.add_warning(
                        "SBTI-EC-004",
                        "Total Scope 3 emissions is zero; cannot assess materiality",
                        ComplianceSeverity.LOW,
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SBTI-EC-004",
                    "Could not calculate materiality (invalid Scope 3 total)",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SBTI-EC-004",
                "Total Scope 3 emissions not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 7 "
                    "materiality against 1% threshold."
                ),
            )

        # SBTI-EC-005: Reduction initiatives
        reduction_initiatives = (
            result.get("reduction_initiatives")
            or result.get("reduction_actions")
            or result.get("actions")
        )
        if reduction_initiatives:
            state.add_pass(
                "SBTI-EC-005",
                "Reduction initiatives documented",
            )
        else:
            state.add_warning(
                "SBTI-EC-005",
                "Reduction initiatives not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document reduction initiatives for employee commuting "
                    "(e.g., telework expansion, transit subsidies, cycling "
                    "infrastructure, carpooling programs, EV incentives)."
                ),
                regulation_reference="SBTi Monitoring Report Guidance",
            )

        # SBTI-EC-006: Base year recalculation policy
        base_year_policy = (
            result.get("base_year_recalculation_policy")
            or result.get("base_year_policy")
        )
        base_year = result.get("base_year")
        if base_year_policy:
            state.add_pass(
                "SBTI-EC-006",
                "Base year recalculation policy documented",
            )
        elif base_year:
            state.add_warning(
                "SBTI-EC-006",
                "Base year present but recalculation policy not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document the base year recalculation policy. "
                    "Significant structural changes (e.g., remote work "
                    "policy changes, office closures) may trigger "
                    "base year recalculation per SBTi criteria."
                ),
                regulation_reference="SBTi Criteria v5.1, C14",
            )
        else:
            state.add_warning(
                "SBTI-EC-006",
                "Base year and recalculation policy not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document both the base year and recalculation policy."
                ),
                regulation_reference="SBTi Criteria v5.1, C14",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: SB 253
    # ==========================================================================

    def check_sb_253(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253 (Climate Corporate Data
        Accountability Act).

        SB 253 requires third-party assurance readiness for Scope 3.

        Checks:
            - SB253-EC-001: Total CO2e present
            - SB253-EC-002: Methodology documented
            - SB253-EC-003: Assurance opinion available
            - SB253-EC-004: Materiality > 1% threshold
            - SB253-EC-005: Data retention documentation
            - SB253-EC-006: Audit trail / provenance

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

        # SB253-EC-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("SB253-EC-001", "Total CO2e present")
        else:
            state.add_fail(
                "SB253-EC-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 7 emissions for "
                    "SB 253 compliance."
                ),
                regulation_reference="SB 253, Section 38532(a)",
            )

        # SB253-EC-002: Methodology documented
        methodology = (
            result.get("methodology")
            or result.get("method")
            or result.get("calculation_method")
        )
        if methodology:
            state.add_pass("SB253-EC-002", "Methodology documented")
        else:
            state.add_fail(
                "SB253-EC-002",
                "Methodology not documented",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Document the calculation methodology in accordance "
                    "with GHG Protocol standards as required by SB 253."
                ),
                regulation_reference="SB 253, Section 38532(b)",
            )

        # SB253-EC-003: Assurance opinion available
        assurance = (
            result.get("assurance_opinion")
            or result.get("assurance")
            or result.get("verification_status")
        )
        if assurance is not None:
            state.add_pass("SB253-EC-003", "Assurance opinion available")
        else:
            state.add_warning(
                "SB253-EC-003",
                "Assurance opinion not available",
                ComplianceSeverity.HIGH,
                recommendation=(
                    "Obtain limited or reasonable assurance opinion for "
                    "Scope 3 emissions. SB 253 requires independent "
                    "third-party assurance starting 2030. Employee "
                    "commuting data should be assurance-ready."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        # SB253-EC-004: Materiality > 1% threshold
        total_scope3 = result.get("total_scope3_co2e")
        if total_co2e and total_scope3:
            try:
                cat7_decimal = Decimal(str(total_co2e))
                scope3_decimal = Decimal(str(total_scope3))
                if scope3_decimal > 0:
                    cat7_pct = (
                        cat7_decimal / scope3_decimal * Decimal("100")
                    )
                    if cat7_pct > Decimal("1"):
                        state.add_pass(
                            "SB253-EC-004",
                            (
                                f"Category 7 exceeds 1% materiality "
                                f"({cat7_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                        )
                    else:
                        state.add_warning(
                            "SB253-EC-004",
                            (
                                f"Category 7 below 1% materiality threshold "
                                f"({cat7_pct.quantize(_QUANT_2DP, rounding=ROUNDING)}% of Scope 3)"
                            ),
                            ComplianceSeverity.LOW,
                            recommendation=(
                                "Category 7 is below 1% of total Scope 3. "
                                "Consider whether separate reporting is warranted."
                            ),
                        )
                else:
                    state.add_warning(
                        "SB253-EC-004",
                        "Total Scope 3 is zero; cannot assess materiality",
                        ComplianceSeverity.LOW,
                    )
            except (InvalidOperation, ZeroDivisionError):
                state.add_warning(
                    "SB253-EC-004",
                    "Could not calculate materiality",
                    ComplianceSeverity.LOW,
                )
        else:
            state.add_warning(
                "SB253-EC-004",
                "Total Scope 3 not provided for materiality assessment",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Provide total Scope 3 emissions to assess "
                    "Category 7 materiality against 1% threshold."
                ),
            )

        # SB253-EC-005: Data retention documentation
        data_retention = (
            result.get("data_retention_policy")
            or result.get("data_retention")
            or result.get("record_keeping")
        )
        if data_retention:
            state.add_pass("SB253-EC-005", "Data retention documented")
        else:
            state.add_warning(
                "SB253-EC-005",
                "Data retention policy not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document data retention period for emission "
                    "calculations and source data. SB 253 requires "
                    "data to be available for verification."
                ),
                regulation_reference="SB 253, Section 38532(c)",
            )

        # SB253-EC-006: Audit trail / provenance
        provenance_hash = (
            result.get("provenance_hash")
            or result.get("audit_trail")
            or result.get("provenance")
        )
        if provenance_hash:
            state.add_pass("SB253-EC-006", "Audit trail / provenance present")
        else:
            state.add_warning(
                "SB253-EC-006",
                "Audit trail / provenance hash not present",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Include SHA-256 provenance hash for complete "
                    "audit trail. This supports third-party assurance."
                ),
                regulation_reference="SB 253, Section 38532(d)",
            )

        return state.to_result()

    # ==========================================================================
    # Framework: GRI 305
    # ==========================================================================

    def check_gri_305(self, result: dict) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305 Emissions Standard.

        Checks:
            - GRI-EC-001: Total CO2e in metric tonnes
            - GRI-EC-002: Gases included documented
            - GRI-EC-003: Base year present
            - GRI-EC-004: Standards referenced
            - GRI-EC-005: Emission factor sources
            - GRI-EC-006: Consolidation approach
            - GRI-EC-007: Intensity ratios

        Args:
            result: Calculation result dictionary.

        Returns:
            ComplianceCheckResult for GRI.

        Example:
            >>> res = engine.check_gri_305(calc_result)
            >>> res.framework.value
            'gri'
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GRI)

        # GRI-EC-001: Total emissions present
        total_co2e = result.get("total_co2e")
        if total_co2e is not None and Decimal(str(total_co2e)) > 0:
            state.add_pass("GRI-EC-001", "Total CO2e reported")
        else:
            state.add_fail(
                "GRI-EC-001",
                "Total CO2e missing or zero",
                ComplianceSeverity.CRITICAL,
                recommendation=(
                    "Report total Category 7 emissions in metric tonnes "
                    "CO2e per GRI 305-3."
                ),
                regulation_reference="GRI 305-3",
            )

        # GRI-EC-002: Gases included
        gases = (
            result.get("gases_included")
            or result.get("emission_gases")
            or result.get("gases")
        )
        if gases:
            state.add_pass("GRI-EC-002", "Gases included documented")
        else:
            state.add_warning(
                "GRI-EC-002",
                "Gases included not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document which GHGs are included in the calculation "
                    "(CO2, CH4, N2O, or CO2e aggregate). Employee "
                    "commuting typically reports CO2e aggregate."
                ),
                regulation_reference="GRI 305-3(c)",
            )

        # GRI-EC-003: Base year
        base_year = result.get("base_year")
        if base_year is not None:
            state.add_pass("GRI-EC-003", "Base year present")
        else:
            state.add_warning(
                "GRI-EC-003",
                "Base year not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document the base year and rationale for choosing it. "
                    "GRI requires base year disclosure for trend reporting. "
                    "Consider COVID-19 era impacts on commuting patterns."
                ),
                regulation_reference="GRI 305-5(a)",
            )

        # GRI-EC-004: Standards referenced
        standards = (
            result.get("standards_used")
            or result.get("standards")
            or result.get("framework_references")
        )
        if standards:
            state.add_pass("GRI-EC-004", "Standards referenced")
        else:
            state.add_warning(
                "GRI-EC-004",
                "Standards used not referenced",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Reference the standards and methodologies used "
                    "(e.g., GHG Protocol Scope 3 Category 7, "
                    "DEFRA conversion factors, IEA grid factors)."
                ),
                regulation_reference="GRI 305-3(e)",
            )

        # GRI-EC-005: Source of emission factors
        ef_sources = result.get("ef_sources") or result.get("ef_source")
        if ef_sources:
            state.add_pass("GRI-EC-005", "Emission factor sources documented")
        else:
            state.add_warning(
                "GRI-EC-005",
                "Emission factor sources not documented",
                ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Document emission factor sources and publication year "
                    "(e.g., DEFRA 2024, EPA 2024, IEA 2024)."
                ),
                regulation_reference="GRI 305-3(d)",
            )

        # GRI-EC-006: Consolidation approach
        consolidation = (
            result.get("consolidation_approach")
            or result.get("org_boundary")
            or result.get("boundary_approach")
        )
        if consolidation:
            state.add_pass("GRI-EC-006", "Consolidation approach documented")
        else:
            state.add_warning(
                "GRI-EC-006",
                "Consolidation approach not documented",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Document the consolidation approach "
                    "(equity share, financial control, or operational control)."
                ),
                regulation_reference="GRI 305-1(b)",
            )

        # GRI-EC-007: Intensity ratios
        intensity = (
            result.get("intensity_ratios")
            or result.get("intensity_per_employee")
            or result.get("co2e_per_fte")
        )
        if intensity is not None:
            state.add_pass("GRI-EC-007", "Intensity ratios present")
        else:
            state.add_warning(
                "GRI-EC-007",
                "Emissions intensity ratios not provided",
                ComplianceSeverity.LOW,
                recommendation=(
                    "Calculate and report intensity ratios "
                    "(e.g., tCO2e per employee, tCO2e per FTE) "
                    "for GRI 305-4 compliance."
                ),
                regulation_reference="GRI 305-4",
            )

        return state.to_result()

    # ==========================================================================
    # Double-Counting Prevention (10 Rules)
    # ==========================================================================

    def check_double_counting(
        self, commutes: list
    ) -> List[dict]:
        """
        Validate 10 double-counting prevention rules for employee commuting.

        Rules:
            DC-EC-001: Exclude company-owned/leased vehicles (should be Scope 1)
            DC-EC-002: Exclude business travel trips (should be Category 6)
            DC-EC-003: Exclude company shuttle services (Scope 1 or Cat 4)
            DC-EC-004: Telework energy not double-counted with Scope 2
            DC-EC-005: No overlap with Category 6 business travel
            DC-EC-006: Avoid commute counted as Category 4 freight
            DC-EC-007: Active transport zero emissions (not negative)
            DC-EC-008: WTT not double-counted with Category 3
            DC-EC-009: Shuttle to transit hub allocation
            DC-EC-010: EV charging at office not in Category 7

        Args:
            commutes: List of commute dictionaries, each containing fields
                like mode, vehicle_ownership, trip_type, includes_business,
                telework_scope2_included, wtt_separately_reported, etc.

        Returns:
            List of finding dictionaries with rule_code, description,
            severity, and affected commute indices.

        Example:
            >>> findings = engine.check_double_counting(commutes)
            >>> len(findings)
            0  # No double-counting issues
        """
        findings: List[dict] = []

        for idx, commute in enumerate(commutes):
            # DC-EC-001: No company-owned/leased vehicles
            self._check_dc_ec_001(idx, commute, findings)
            # DC-EC-002: No business travel trips
            self._check_dc_ec_002(idx, commute, findings)
            # DC-EC-003: No company shuttle services
            self._check_dc_ec_003(idx, commute, findings)
            # DC-EC-004: Telework energy not in Scope 2
            self._check_dc_ec_004(idx, commute, findings)
            # DC-EC-005: No overlap with Category 6
            self._check_dc_ec_005(idx, commute, findings)
            # DC-EC-006: No commute as Category 4 freight
            self._check_dc_ec_006(idx, commute, findings)
            # DC-EC-007: Active transport zero emissions
            self._check_dc_ec_007(idx, commute, findings)
            # DC-EC-008: WTT not double-counted with Category 3
            self._check_dc_ec_008(idx, commute, findings)
            # DC-EC-009: Shuttle to transit hub allocation
            self._check_dc_ec_009(idx, commute, findings)
            # DC-EC-010: EV charging at office
            self._check_dc_ec_010(idx, commute, findings)

        logger.info(
            "Double-counting check: %d commutes, %d findings",
            len(commutes),
            len(findings),
        )

        return findings

    def _check_dc_ec_001(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-001: Exclude company-owned/leased vehicles.

        Company-owned or leased vehicles should be reported under Scope 1
        (mobile combustion), not Category 7 (Employee Commuting).

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        vehicle_ownership = commute.get("vehicle_ownership", "").lower()
        if vehicle_ownership in (
            "company_owned", "company", "fleet",
            "company_leased", "leased", "corporate",
        ):
            findings.append({
                "rule_code": "DC-EC-001",
                "description": (
                    f"Commute {idx}: Company-owned/leased vehicle detected. "
                    "Should be reported under Scope 1 (mobile combustion), "
                    "not Category 7."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "Exclude company-owned/leased vehicles from Category 7. "
                    "Report under Scope 1 (mobile combustion)."
                ),
            })

    def _check_dc_ec_002(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-002: Exclude business travel trips.

        Business travel trips should be reported under Category 6,
        not Category 7 (Employee Commuting).

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        purpose = commute.get("purpose", "").lower()
        trip_type = commute.get("trip_type", "").lower()
        if purpose in (
            "business_travel", "business", "conference",
            "client_visit", "training_offsite", "site_visit",
        ) or trip_type in ("business_travel", "business"):
            findings.append({
                "rule_code": "DC-EC-002",
                "description": (
                    f"Commute {idx}: Business travel trip detected. "
                    "Should be reported under Category 6 (Business Travel), "
                    "not Category 7."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.CATEGORY_6.value,
                "recommendation": (
                    "Exclude business travel trips from Category 7. "
                    "Report under Category 6 (Business Travel)."
                ),
            })

    def _check_dc_ec_003(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-003: Exclude company shuttle services.

        Company-operated shuttle services should be reported under Scope 1
        (if company-owned vehicles) or Category 4 (if contracted).

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        mode = commute.get("mode", "").lower()
        is_company_operated = commute.get("company_operated", False)
        shuttle_ownership = commute.get("shuttle_ownership", "").lower()

        if mode in COMPANY_TRANSPORT_MODES or (
            mode == "shuttle" and is_company_operated
        ):
            if shuttle_ownership in (
                "company_owned", "company", "fleet",
            ) or is_company_operated:
                findings.append({
                    "rule_code": "DC-EC-003",
                    "description": (
                        f"Commute {idx}: Company-operated shuttle detected. "
                        "Company-owned shuttle should be reported under "
                        "Scope 1; contracted shuttle may be Category 4."
                    ),
                    "severity": ComplianceSeverity.HIGH.value,
                    "commute_index": idx,
                    "category": DoubleCountingCategory.SCOPE_1.value,
                    "recommendation": (
                        "Classify shuttle services: company-owned goes to "
                        "Scope 1; third-party contracted may go to "
                        "Category 4 (upstream transportation). "
                        "Only employee-paid shuttles belong in Category 7."
                    ),
                })

    def _check_dc_ec_004(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-004: Telework energy not double-counted with Scope 2.

        Telework home-office energy use must not be double-counted
        with the company's Scope 2 (purchased electricity) if the
        employee charges telework energy costs to the company.

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        mode = commute.get("mode", "").lower()
        if mode != "telework" and mode != "remote_work":
            return

        telework_in_scope2 = commute.get("telework_scope2_included", False)
        telework_reimbursed = commute.get("energy_reimbursed", False)

        if telework_in_scope2:
            findings.append({
                "rule_code": "DC-EC-004",
                "description": (
                    f"Commute {idx}: Telework energy already included in "
                    "Scope 2. Must not be double-counted in Category 7."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.SCOPE_2.value,
                "recommendation": (
                    "If telework energy is already accounted in Scope 2 "
                    "(e.g., company pays the electricity bill for home "
                    "offices), exclude from Category 7 to prevent "
                    "double-counting."
                ),
            })

        if telework_reimbursed:
            # Energy reimbursement could indicate Scope 2 overlap
            if not telework_in_scope2:
                findings.append({
                    "rule_code": "DC-EC-004",
                    "description": (
                        f"Commute {idx}: Telework energy is reimbursed by "
                        "company. Verify this is not also in Scope 2."
                    ),
                    "severity": ComplianceSeverity.MEDIUM.value,
                    "commute_index": idx,
                    "category": DoubleCountingCategory.SCOPE_2.value,
                    "recommendation": (
                        "When the company reimburses home-office energy, "
                        "verify whether it is already captured in Scope 2 "
                        "before including in Category 7."
                    ),
                })

    def _check_dc_ec_005(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-005: No overlap with Category 6 business travel.

        Ensure commute records are not also reported in Category 6.

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        reported_in_cat6 = commute.get("reported_in_cat6", False)
        also_in_cat6 = commute.get("also_in_business_travel", False)

        if reported_in_cat6 or also_in_cat6:
            findings.append({
                "rule_code": "DC-EC-005",
                "description": (
                    f"Commute {idx}: Also reported in Category 6 "
                    "(Business Travel). Double-counting risk."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.CATEGORY_6.value,
                "recommendation": (
                    "Ensure the trip is reported in only one category. "
                    "Regular commuting goes in Category 7; business "
                    "trips go in Category 6."
                ),
            })

    def _check_dc_ec_006(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-006: Avoid commute counted as Category 4 freight.

        Employee commute trips must not be classified as upstream
        transportation (freight/cargo).

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        trip_type = commute.get("trip_type", "").lower()
        reported_in_cat4 = commute.get("reported_in_cat4", False)

        if trip_type in ("freight", "cargo", "logistics", "delivery"):
            findings.append({
                "rule_code": "DC-EC-006",
                "description": (
                    f"Commute {idx}: Freight/cargo trip type detected in "
                    "Category 7. This should be Category 4 or 9."
                ),
                "severity": ComplianceSeverity.CRITICAL.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.CATEGORY_4.value,
                "recommendation": (
                    "Freight/cargo transportation does not belong in "
                    "Category 7 (Employee Commuting). Report under "
                    "Category 4 (upstream) or Category 9 (downstream)."
                ),
            })

        if reported_in_cat4:
            findings.append({
                "rule_code": "DC-EC-006",
                "description": (
                    f"Commute {idx}: Also reported in Category 4 "
                    "(Upstream Transportation). Double-counting risk."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.CATEGORY_4.value,
                "recommendation": (
                    "Ensure commute is only in Category 7. "
                    "Freight goes in Category 4."
                ),
            })

    def _check_dc_ec_007(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-007: Active transport zero emissions (not negative).

        Walking and cycling must have zero emissions. Any negative
        value is invalid (no carbon credits for active transport).

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        mode = commute.get("mode", "").lower()
        if mode not in ACTIVE_TRANSPORT_MODES:
            return

        emissions = commute.get("emissions") or commute.get("co2e")
        if emissions is not None:
            try:
                emissions_decimal = Decimal(str(emissions))
                if emissions_decimal < Decimal("0"):
                    findings.append({
                        "rule_code": "DC-EC-007",
                        "description": (
                            f"Commute {idx}: Active transport ({mode}) has "
                            f"negative emissions ({emissions_decimal}). "
                            "Active transport must have zero emissions."
                        ),
                        "severity": ComplianceSeverity.HIGH.value,
                        "commute_index": idx,
                        "recommendation": (
                            "Active transport (cycling, walking) must have "
                            "zero emissions. Negative values (carbon credits) "
                            "are not permitted under GHG Protocol."
                        ),
                    })
                elif emissions_decimal > Decimal("0"):
                    findings.append({
                        "rule_code": "DC-EC-007",
                        "description": (
                            f"Commute {idx}: Active transport ({mode}) has "
                            f"positive emissions ({emissions_decimal}). "
                            "Walking and cycling should be zero."
                        ),
                        "severity": ComplianceSeverity.MEDIUM.value,
                        "commute_index": idx,
                        "recommendation": (
                            "Walking and pedal cycling have zero direct "
                            "emissions. Remove any assigned emission factor. "
                            "Note: e-bikes/e-scooters do have small emissions."
                        ),
                    })
            except (InvalidOperation, ValueError):
                pass  # Non-numeric emissions handled elsewhere

    def _check_dc_ec_008(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-008: WTT not double-counted with Category 3.

        Well-to-tank (WTT) emissions should be reported consistently.
        If WTT is included in Category 7 and also reported separately
        in Category 3 (Fuel & Energy Activities), it creates double counting.

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        wtt_included = commute.get("wtt_included", False)
        wtt_separately_reported = commute.get("wtt_separately_reported", False)
        wtt_in_cat3 = commute.get("wtt_reported_in_cat3", False)

        if wtt_included and wtt_in_cat3:
            findings.append({
                "rule_code": "DC-EC-008",
                "description": (
                    f"Commute {idx}: WTT emissions included in Category 7 "
                    "AND reported in Category 3. Double-counting risk."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.CATEGORY_3.value,
                "recommendation": (
                    "Report WTT in either Category 3 "
                    "(Fuel & Energy Activities) or include as a "
                    "component in Category 7, not both. Document "
                    "the WTT boundary allocation clearly."
                ),
            })

        if wtt_separately_reported and wtt_in_cat3:
            findings.append({
                "rule_code": "DC-EC-008",
                "description": (
                    f"Commute {idx}: WTT separately reported AND "
                    "also in Category 3. Confirm no overlap."
                ),
                "severity": ComplianceSeverity.MEDIUM.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.CATEGORY_3.value,
                "recommendation": (
                    "Verify WTT boundary: if WTT is reported as a "
                    "memo item in Category 7, ensure it is excluded "
                    "from Category 3 totals."
                ),
            })

    def _check_dc_ec_009(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-009: Shuttle to transit hub allocation.

        If an employee takes a company shuttle to a transit hub
        (e.g., park-and-ride), the shuttle portion must be correctly
        allocated to avoid counting it in both Category 7 and Scope 1/Cat 4.

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        is_multimodal = commute.get("multimodal", False)
        segments = commute.get("segments") or commute.get("legs") or []
        has_shuttle_segment = False
        shuttle_company_owned = False

        if isinstance(segments, list):
            for seg in segments:
                seg_mode = (
                    seg.get("mode", "").lower() if isinstance(seg, dict)
                    else ""
                )
                if seg_mode in ("shuttle", "company_shuttle", "company_bus"):
                    has_shuttle_segment = True
                    seg_ownership = (
                        seg.get("ownership", "").lower()
                        if isinstance(seg, dict) else ""
                    )
                    if seg_ownership in (
                        "company_owned", "company", "fleet",
                    ):
                        shuttle_company_owned = True

        if has_shuttle_segment and shuttle_company_owned:
            findings.append({
                "rule_code": "DC-EC-009",
                "description": (
                    f"Commute {idx}: Multi-modal trip includes a "
                    "company-owned shuttle segment to transit hub. "
                    "The shuttle portion should be Scope 1, not Cat 7."
                ),
                "severity": ComplianceSeverity.MEDIUM.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "For multi-modal commutes with a company shuttle "
                    "to a transit hub, allocate the shuttle segment to "
                    "Scope 1 (if company-owned) and only the remaining "
                    "segments to Category 7."
                ),
            })

        # Also flag if the shuttle leg has its own emissions already
        # counted elsewhere
        mode = commute.get("mode", "").lower()
        shuttle_in_scope1 = commute.get("shuttle_in_scope1", False)
        if mode == "shuttle" and shuttle_in_scope1:
            findings.append({
                "rule_code": "DC-EC-009",
                "description": (
                    f"Commute {idx}: Shuttle emissions already counted "
                    "in Scope 1. Exclude from Category 7."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.SCOPE_1.value,
                "recommendation": (
                    "If shuttle emissions are in Scope 1, do not "
                    "also include them in Category 7."
                ),
            })

    def _check_dc_ec_010(
        self, idx: int, commute: dict, findings: List[dict]
    ) -> None:
        """
        DC-EC-010: EV charging at office not in Category 7.

        If an employee charges their electric vehicle at the office,
        the electricity consumption is part of the company's Scope 2
        (purchased electricity), not Category 7.

        Args:
            idx: Index of the commute record.
            commute: Commute record dictionary.
            findings: Findings list to append to (mutated in place).
        """
        mode = commute.get("mode", "").lower()
        vehicle_type = commute.get("vehicle_type", "").lower()
        charging_location = commute.get("charging_location", "").lower()

        is_ev = (
            mode in ELECTRIC_MODES
            or vehicle_type in ("bev", "ev", "electric", "plugin_hybrid")
        )

        if is_ev and charging_location in (
            "office", "workplace", "company", "onsite",
        ):
            findings.append({
                "rule_code": "DC-EC-010",
                "description": (
                    f"Commute {idx}: EV charging at office/workplace "
                    "detected. Office EV charging electricity is Scope 2, "
                    "not Category 7."
                ),
                "severity": ComplianceSeverity.HIGH.value,
                "commute_index": idx,
                "category": DoubleCountingCategory.SCOPE_2.value,
                "recommendation": (
                    "EV charging electricity consumed at the office "
                    "is part of the company's Scope 2 (purchased "
                    "electricity). Only the driving emissions using "
                    "home-charged electricity belong in Category 7. "
                    "Allocate EV emissions proportionally between "
                    "home and office charging."
                ),
            })

    # ==========================================================================
    # Category Boundary
    # ==========================================================================

    def check_category_boundary(
        self, commute: dict
    ) -> Dict[str, Any]:
        """
        Determine if a commute belongs in Category 7, Category 6,
        Scope 1, Scope 2, or other.

        Classification Rules:
            - Company-owned vehicle -> Scope 1
            - Business travel -> Category 6
            - Company shuttle (owned) -> Scope 1
            - EV charging at office -> Scope 2
            - Freight/cargo -> Category 4 or 9
            - Regular commute (third-party vehicle) -> Category 7
            - Telework -> Category 7 (unless Scope 2 overlap)
            - All other -> Excluded

        Args:
            commute: Commute dictionary with mode, purpose,
                vehicle_ownership, trip_type fields.

        Returns:
            Dictionary with classification, reason, and any warnings.

        Example:
            >>> result = engine.check_category_boundary({
            ...     "mode": "sov",
            ...     "purpose": "commuting",
            ...     "vehicle_ownership": "personal"
            ... })
            >>> result["classification"]
            'CATEGORY_7'
        """
        vehicle_ownership = commute.get("vehicle_ownership", "").lower()
        purpose = commute.get("purpose", "").lower()
        trip_type = commute.get("trip_type", "").lower()
        mode = commute.get("mode", "").lower()

        warnings: List[str] = []

        # Rule 1: Company-owned vehicles -> Scope 1
        if vehicle_ownership in (
            "company_owned", "company", "fleet",
            "company_leased", "leased", "corporate",
        ):
            return {
                "classification": BoundaryClassification.SCOPE_1.value,
                "reason": (
                    "Company-owned/leased vehicle emissions belong in "
                    "Scope 1 (mobile combustion), not Category 7."
                ),
                "warnings": warnings,
                "commute": commute,
            }

        # Rule 2: Business travel -> Category 6
        if purpose in (
            "business_travel", "business", "conference",
            "client_visit", "training_offsite",
        ) or trip_type in ("business_travel", "business"):
            return {
                "classification": BoundaryClassification.CATEGORY_6.value,
                "reason": (
                    "Business travel belongs in Category 6, "
                    "not Category 7 (Employee Commuting)."
                ),
                "warnings": warnings,
                "commute": commute,
            }

        # Rule 3: Company shuttle (owned) -> Scope 1
        if mode in COMPANY_TRANSPORT_MODES:
            company_operated = commute.get("company_operated", False)
            if company_operated:
                return {
                    "classification": BoundaryClassification.SCOPE_1.value,
                    "reason": (
                        "Company-operated shuttle emissions belong in "
                        "Scope 1, not Category 7."
                    ),
                    "warnings": warnings,
                    "commute": commute,
                }

        # Rule 4: Freight/cargo -> Category 4 or 9
        if trip_type in ("freight", "cargo", "logistics", "delivery"):
            classification = BoundaryClassification.CATEGORY_4.value
            if commute.get("direction", "").lower() == "downstream":
                classification = BoundaryClassification.EXCLUDED.value
            return {
                "classification": classification,
                "reason": (
                    "Freight/cargo transportation does not belong in "
                    "Category 7 (Employee Commuting)."
                ),
                "warnings": warnings,
                "commute": commute,
            }

        # Rule 5: Telework -> Category 7 (with Scope 2 warning)
        if mode in ("telework", "remote_work", "work_from_home"):
            telework_in_scope2 = commute.get(
                "telework_scope2_included", False
            )
            if telework_in_scope2:
                warnings.append(
                    "Telework energy is already in Scope 2. "
                    "Verify no double-counting."
                )
            return {
                "classification": BoundaryClassification.CATEGORY_7.value,
                "reason": (
                    "Telework/remote work home-office energy belongs "
                    "in Category 7."
                ),
                "warnings": warnings,
                "commute": commute,
            }

        # Rule 6: EV charging at office -> Scope 2 warning
        charging_location = commute.get("charging_location", "").lower()
        vehicle_type = commute.get("vehicle_type", "").lower()
        is_ev = vehicle_type in ("bev", "ev", "electric", "plugin_hybrid")
        if is_ev and charging_location in ("office", "workplace", "onsite"):
            warnings.append(
                "EV charging at office is Scope 2. Only home-charged "
                "driving emissions belong in Category 7."
            )

        # Rule 7: Regular commute -> Category 7
        valid_commute_purposes = {
            "commuting", "commute", "home_to_work",
            "work_to_home", "", "regular",
        }
        if purpose in valid_commute_purposes or not purpose:
            return {
                "classification": BoundaryClassification.CATEGORY_7.value,
                "reason": (
                    "Regular employee commute using third-party "
                    "or personal transportation belongs in Category 7."
                ),
                "warnings": warnings,
                "commute": commute,
            }

        # Default: Excluded
        return {
            "classification": BoundaryClassification.EXCLUDED.value,
            "reason": (
                f"Commute with purpose '{purpose}' and type '{trip_type}' "
                "could not be classified into a Scope 3 category."
            ),
            "warnings": warnings,
            "commute": commute,
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
                "info_findings": [],
                "total_findings": 0,
                "recommendations": [],
                "provenance_hash": "",
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
        info_findings: List[dict] = []

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
                elif severity == ComplianceSeverity.INFO.value:
                    info_findings.append(finding_with_fw)

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
            + len(info_findings)
        )

        # Calculate provenance hash for the summary
        provenance_input = (
            f"{float(overall_score)}:{overall_status.value}:"
            f"{total_findings}:{len(results)}"
        )
        provenance_hash = calculate_provenance_hash(provenance_input)

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
            "recommendations": all_recommendations,
            "provenance_hash": provenance_hash,
        }

    # ==========================================================================
    # Engine Statistics
    # ==========================================================================

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Return engine statistics.

        Returns:
            Dictionary with engine_id, version, check_count,
            enabled frameworks, and configuration state.

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
            "telework_disclosure_required": (
                self._config.compliance.telework_disclosure_required
            ),
            "mode_share_required": (
                self._config.compliance.mode_share_required
            ),
            "double_counting_check": (
                self._config.compliance.double_counting_check
            ),
            "boundary_enforcement": (
                self._config.compliance.boundary_enforcement
            ),
        }

    # ==========================================================================
    # Batch Compliance Check
    # ==========================================================================

    def check_batch_compliance(
        self,
        results: List[dict],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run compliance checks across a batch of calculation results.

        Aggregates individual results and produces a consolidated
        compliance summary with batch-level statistics.

        Args:
            results: List of calculation result dictionaries.
            frameworks: Optional list of framework names to check.

        Returns:
            Dictionary with batch_size, per-result findings,
            aggregated scores, and overall batch compliance.

        Example:
            >>> batch_result = engine.check_batch_compliance(
            ...     [calc_result_1, calc_result_2]
            ... )
            >>> batch_result["batch_size"]
            2
        """
        start_time = time.monotonic()
        batch_size = len(results)

        logger.info(
            "Running batch compliance check: %d results",
            batch_size,
        )

        per_result_checks: List[Dict[str, Any]] = []
        all_framework_results: Dict[str, List[ComplianceCheckResult]] = {}

        for i, calc_result in enumerate(results):
            try:
                fw_results = self.check_all_frameworks(
                    calc_result, frameworks=frameworks
                )
                summary = self.get_compliance_summary(fw_results)
                per_result_checks.append({
                    "index": i,
                    "status": "success",
                    "overall_score": summary["overall_score"],
                    "overall_status": summary["overall_status"],
                    "total_findings": summary["total_findings"],
                })

                for fw_name, fw_result in fw_results.items():
                    if fw_name not in all_framework_results:
                        all_framework_results[fw_name] = []
                    all_framework_results[fw_name].append(fw_result)

            except Exception as e:
                logger.error(
                    "Batch compliance check failed for result %d: %s",
                    i, str(e), exc_info=True,
                )
                per_result_checks.append({
                    "index": i,
                    "status": "error",
                    "error": str(e),
                    "overall_score": 0.0,
                    "overall_status": ComplianceStatus.FAIL.value,
                    "total_findings": 0,
                })

        # Aggregate framework scores
        aggregated_scores: Dict[str, Dict[str, Any]] = {}
        for fw_name, fw_result_list in all_framework_results.items():
            scores = [r.score for r in fw_result_list]
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                aggregated_scores[fw_name] = {
                    "average_score": float(
                        avg_score.quantize(_QUANT_2DP, rounding=ROUNDING)
                    ),
                    "min_score": float(min_score),
                    "max_score": float(max_score),
                    "count": len(scores),
                }

        # Overall batch statistics
        successful_results = [
            r for r in per_result_checks if r["status"] == "success"
        ]
        batch_scores = [r["overall_score"] for r in successful_results]
        batch_avg = (
            sum(batch_scores) / len(batch_scores) if batch_scores else 0.0
        )

        duration = time.monotonic() - start_time

        logger.info(
            "Batch compliance check complete: %d results, "
            "avg_score=%.2f, duration=%.4fs",
            batch_size, batch_avg, duration,
        )

        return {
            "batch_size": batch_size,
            "successful_checks": len(successful_results),
            "failed_checks": batch_size - len(successful_results),
            "average_score": round(batch_avg, 2),
            "per_result_checks": per_result_checks,
            "aggregated_framework_scores": aggregated_scores,
            "duration_seconds": round(duration, 4),
        }

    # ==========================================================================
    # Required Disclosures
    # ==========================================================================

    def get_required_disclosures(
        self,
        framework: str,
    ) -> List[str]:
        """
        Get the list of required disclosures for a specific framework.

        Args:
            framework: Framework name (e.g., "ghg_protocol", "csrd_esrs").

        Returns:
            List of required disclosure field names.

        Example:
            >>> disclosures = engine.get_required_disclosures("ghg_protocol")
            >>> "total_co2e" in disclosures
            True
        """
        try:
            fw_enum = ComplianceFramework(framework)
        except ValueError:
            # Try uppercase mapping
            fw_map = {
                "GHG_PROTOCOL_SCOPE3": ComplianceFramework.GHG_PROTOCOL,
                "ISO_14064": ComplianceFramework.ISO_14064,
                "CSRD_ESRS_E1": ComplianceFramework.CSRD_ESRS,
                "CDP": ComplianceFramework.CDP,
                "SBTI": ComplianceFramework.SBTI,
                "SB_253": ComplianceFramework.SB_253,
                "GRI_305": ComplianceFramework.GRI,
            }
            fw_enum = fw_map.get(framework)
            if fw_enum is None:
                logger.warning(
                    "Unknown framework '%s' for required disclosures",
                    framework,
                )
                return []

        return FRAMEWORK_REQUIRED_DISCLOSURES.get(fw_enum, [])

    def check_disclosure_completeness(
        self,
        result: dict,
        framework: str,
    ) -> Dict[str, Any]:
        """
        Check whether a result contains all required disclosures
        for a given framework.

        Args:
            result: Calculation result dictionary.
            framework: Framework name.

        Returns:
            Dictionary with present, missing, and completeness_pct fields.

        Example:
            >>> completeness = engine.check_disclosure_completeness(
            ...     calc_result, "ghg_protocol"
            ... )
            >>> completeness["completeness_pct"]
            83.3
        """
        required = self.get_required_disclosures(framework)
        if not required:
            return {
                "framework": framework,
                "required": [],
                "present": [],
                "missing": [],
                "completeness_pct": 100.0,
            }

        present = []
        missing = []

        for field_name in required:
            value = result.get(field_name)
            if value is not None and value != "" and value != []:
                present.append(field_name)
            else:
                missing.append(field_name)

        total = len(required)
        pct = (len(present) / total * 100) if total > 0 else 100.0

        return {
            "framework": framework,
            "required": required,
            "present": present,
            "missing": missing,
            "completeness_pct": round(pct, 1),
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
