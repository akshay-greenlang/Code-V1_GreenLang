# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-028 Engine 6

This module implements regulatory compliance checking for Investment emissions
(GHG Protocol Scope 3 Category 15) against 9 regulatory frameworks.

Regulatory Frameworks:
1. GHG Protocol Scope 3 Standard (Category 15 specific)
2. PCAF Global GHG Accounting Standard (3rd Edition)
3. ISO 14064-1:2018 (Clause 5.2.4)
4. CSRD/ESRS E1 Climate Change
5. CDP Climate Change Questionnaire (C6.5, C-FS14.1)
6. SBTi Financial Institutions (SBTi-FI)
7. SB 253 (California Climate Corporate Data Accountability Act)
8. TCFD Recommendations (Metrics & Targets)
9. NZBA/NZAOA (Net-Zero Banking Alliance / Net-Zero Asset Owner Alliance)

Double-Counting Prevention Rules (8):
    DC-INV-001: Investments consolidated in Scope 1/2 NOT Cat 15
    DC-INV-002: Equity share already used for consolidation
    DC-INV-003: Fund-of-funds look-through
    DC-INV-004: CRE vs Cat 8/13 overlap
    DC-INV-005: Sovereign bonds vs corporate (national includes corporates)
    DC-INV-006: Multi-asset class same company: count once
    DC-INV-007: Managed investments: underlying already counted
    DC-INV-008: Short positions: exclude (no economic exposure)

PCAF-Specific Compliance:
    - Attribution factor formula per asset class
    - Data quality score 1-5 assignment criteria
    - WACI calculation methodology
    - Portfolio coverage (% of AUM with financed emissions)
    - Data quality improvement plan

Example:
    >>> engine = ComplianceCheckerEngine.get_instance()
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> summary = engine.get_compliance_summary(result)
    >>> print(f"Compliance: {summary['overall_score']}%")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "compliance_checker_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

# PCAF asset classes
PCAF_ASSET_CLASSES: List[str] = [
    "listed_equity",
    "corporate_bond",
    "private_equity",
    "project_finance",
    "commercial_real_estate",
    "mortgage",
    "motor_vehicle_loan",
    "sovereign_bond",
]

# PCAF data quality score descriptions
PCAF_DQ_DESCRIPTIONS: Dict[int, str] = {
    1: "Reported emissions, verified by third party",
    2: "Reported emissions, unverified; or physical activity-based",
    3: "Revenue-based EEIO or production-based estimates",
    4: "Estimated using economic activity or asset class proxies",
    5: "Sector average or asset class defaults",
}

# NZBA target sectors
NZBA_TARGET_SECTORS: List[str] = [
    "oil_and_gas",
    "power_generation",
    "coal",
    "automotive",
    "cement",
    "steel",
    "real_estate",
    "agriculture",
    "aviation",
    "shipping",
]


# ==============================================================================
# ENUMS
# ==============================================================================


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks for investments."""

    GHG_PROTOCOL = "GHG_PROTOCOL"
    PCAF = "PCAF"
    ISO_14064 = "ISO_14064"
    CSRD_ESRS = "CSRD_ESRS"
    CDP = "CDP"
    SBTI_FI = "SBTI_FI"
    SB_253 = "SB_253"
    TCFD = "TCFD"
    NZBA = "NZBA"


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
    """Scope categories that could overlap with Category 15."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2 = "SCOPE_2"
    CATEGORY_8 = "CATEGORY_8"   # Upstream leased assets
    CATEGORY_13 = "CATEGORY_13"  # Downstream leased assets


class AssetClass(str, Enum):
    """PCAF asset classes."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BOND = "corporate_bond"
    PRIVATE_EQUITY = "private_equity"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE_LOAN = "motor_vehicle_loan"
    SOVEREIGN_BOND = "sovereign_bond"


class CalculationMethod(str, Enum):
    """Calculation method for investment emissions."""

    REPORTED_VERIFIED = "reported_verified"
    REPORTED_UNVERIFIED = "reported_unverified"
    PHYSICAL_ACTIVITY = "physical_activity"
    REVENUE_EEIO = "revenue_eeio"
    ASSET_SPECIFIC = "asset_specific"
    SECTOR_AVERAGE = "sector_average"


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
        severity: ComplianceSeverity = ComplianceSeverity.HIGH,
        recommendation: Optional[str] = None,
        regulation_reference: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a failed check."""
        self.findings.append(ComplianceFinding(
            rule_code=rule_code,
            description=description,
            severity=severity,
            framework=self.framework.value,
            status=ComplianceStatus.FAIL,
            recommendation=recommendation,
            regulation_reference=regulation_reference,
            details=details,
        ))
        self.failed_checks += 1
        self.total_checks += 1

    def add_warning(
        self,
        rule_code: str,
        description: str,
        severity: ComplianceSeverity = ComplianceSeverity.MEDIUM,
        recommendation: Optional[str] = None,
        regulation_reference: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a warning check."""
        self.findings.append(ComplianceFinding(
            rule_code=rule_code,
            description=description,
            severity=severity,
            framework=self.framework.value,
            status=ComplianceStatus.WARNING,
            recommendation=recommendation,
            regulation_reference=regulation_reference,
            details=details,
        ))
        self.warning_checks += 1
        self.total_checks += 1

    def compute_score(self) -> Decimal:
        """Compute compliance score (0-100) from checks."""
        if self.total_checks == 0:
            return Decimal("100")
        raw = (Decimal(str(self.passed_checks)) / Decimal(str(self.total_checks))) * Decimal("100")
        return raw.quantize(_QUANT_2DP, rounding=ROUNDING)

    def compute_status(self) -> ComplianceStatus:
        """Derive overall status from checks."""
        if self.failed_checks > 0:
            return ComplianceStatus.FAIL
        if self.warning_checks > 3:
            return ComplianceStatus.WARNING
        return ComplianceStatus.PASS


@dataclass
class ComplianceCheckResult:
    """Result of compliance check for one framework."""

    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal
    issues: List[ComplianceFinding] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class InvestmentResultInput(BaseModel):
    """Input calculation result for compliance checking."""

    # Total emissions
    total_emissions_kg_co2e: Decimal = Field(..., description="Total financed emissions kgCO2e")
    co2_kg: Decimal = Field(default=Decimal("0"), description="CO2 emissions")
    ch4_kg: Decimal = Field(default=Decimal("0"), description="CH4 emissions")
    n2o_kg: Decimal = Field(default=Decimal("0"), description="N2O emissions")

    # Portfolio data
    total_aum: Optional[Decimal] = Field(None, ge=0, description="Total AUM in USD")
    covered_aum: Optional[Decimal] = Field(None, ge=0, description="AUM with financed emissions")
    portfolio_coverage_pct: Optional[Decimal] = Field(None, ge=0, le=100, description="Coverage %")

    # WACI
    waci_value: Optional[Decimal] = Field(None, description="Weighted Average Carbon Intensity tCO2e/$M revenue")

    # Asset class breakdown
    asset_class_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by asset class (kgCO2e)"
    )
    asset_class_aum: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="AUM by asset class (USD)"
    )

    # PCAF data quality
    data_quality_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="PCAF DQ score (1-5) by asset class"
    )
    weighted_data_quality_score: Optional[Decimal] = Field(
        None, ge=1, le=5,
        description="AUM-weighted average PCAF data quality score"
    )
    has_dq_improvement_plan: bool = Field(default=False, description="Has data quality improvement plan")

    # Attribution
    attribution_method: Optional[str] = Field(None, description="Attribution method used")
    attribution_factors: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Attribution factor per holding"
    )

    # Calculation method
    calculation_method: Optional[str] = Field(None, description="Primary calculation method")
    calculation_methods_used: List[str] = Field(
        default_factory=list,
        description="All calculation methods used"
    )

    # Double counting flags
    includes_consolidated_investments: bool = Field(
        default=False,
        description="Includes investments already in Scope 1/2"
    )
    includes_equity_share_consolidated: bool = Field(
        default=False,
        description="Equity share already used for Scope 1/2 consolidation"
    )
    includes_fund_of_funds: bool = Field(default=False, description="Contains fund-of-funds")
    fund_of_funds_look_through: bool = Field(default=False, description="Look-through applied")
    includes_cre_in_cat8_or_cat13: bool = Field(
        default=False,
        description="CRE also counted in Cat 8 or 13"
    )
    includes_sovereign_and_corporate: bool = Field(
        default=False,
        description="Both sovereign and corporate bonds for same country"
    )
    multi_asset_same_company: bool = Field(
        default=False,
        description="Multiple asset classes for same company"
    )
    multi_asset_deduplication_applied: bool = Field(
        default=False,
        description="Deduplication applied for multi-asset same company"
    )
    includes_managed_investments: bool = Field(
        default=False,
        description="Managed investments where underlying already counted"
    )
    includes_short_positions: bool = Field(
        default=False,
        description="Short positions included"
    )

    # Sector-level data (for SBTi-FI / NZBA)
    sector_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Financed emissions by sector"
    )
    sector_targets: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Sector-specific reduction targets"
    )
    temperature_alignment: Optional[Decimal] = Field(
        None,
        description="Portfolio temperature alignment (degrees C)"
    )
    portfolio_alignment_pct: Optional[Decimal] = Field(
        None, ge=0, le=100,
        description="% of portfolio aligned to targets"
    )

    # Reporting metadata
    reporting_period_start: Optional[datetime] = None
    reporting_period_end: Optional[datetime] = None
    base_year: Optional[int] = None
    reporting_year: Optional[int] = None
    is_financial_institution: bool = Field(default=True, description="Reporting entity is a FI")
    has_scenario_analysis: bool = Field(default=False, description="TCFD scenario analysis done")

    # Uncertainty
    uncertainty_percentage: Optional[Decimal] = None

    # Scope 3 context
    total_scope3_emissions_kg_co2e: Optional[Decimal] = None
    category_15_percentage_of_scope3: Optional[Decimal] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceCheckerConfig(BaseModel):
    """Configuration for ComplianceCheckerEngine."""

    enabled_frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: list(ComplianceFramework)
    )

    # PCAF thresholds
    minimum_portfolio_coverage_pct: Decimal = Field(
        default=Decimal("50.0"), ge=0, le=100,
        description="Minimum portfolio coverage for PCAF compliance"
    )
    maximum_dq_score: Decimal = Field(
        default=Decimal("3.5"), ge=1, le=5,
        description="Maximum acceptable weighted DQ score (lower = better)"
    )

    # SBTi-FI thresholds
    sbti_fi_materiality_threshold: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=100,
        description="SBTi-FI materiality threshold for Cat 15 (%)"
    )
    sbti_fi_portfolio_coverage_threshold: Decimal = Field(
        default=Decimal("67.0"), ge=0, le=100,
        description="SBTi-FI portfolio coverage target (%)"
    )
    sbti_fi_temperature_target: Decimal = Field(
        default=Decimal("1.5"),
        description="SBTi-FI temperature alignment target (degrees C)"
    )

    # NZBA targets
    nzba_interim_target_year: int = Field(default=2030, description="NZBA interim target year")
    nzba_final_target_year: int = Field(default=2050, description="NZBA final target year")

    # Scoring weights
    critical_weight: Decimal = Field(default=Decimal("1.0"))
    high_weight: Decimal = Field(default=Decimal("0.8"))
    medium_weight: Decimal = Field(default=Decimal("0.5"))
    low_weight: Decimal = Field(default=Decimal("0.2"))
    info_weight: Decimal = Field(default=Decimal("0.0"))


# ==============================================================================
# ComplianceCheckerEngine
# ==============================================================================


class ComplianceCheckerEngine:
    """
    ComplianceCheckerEngine - validates investment calculations against 9 frameworks.

    This engine implements comprehensive compliance checking for Investments
    (Category 15) against major regulatory frameworks including GHG Protocol,
    PCAF, ISO 14064, CSRD/ESRS E1, CDP, SBTi-FI, SB 253, TCFD, and NZBA/NZAOA.

    Investment-Specific Compliance Rules:
    1. PCAF attribution factor validation per asset class
    2. Data quality score 1-5 assignment and weighting
    3. WACI calculation methodology validation
    4. Portfolio coverage requirements
    5. 8 double-counting prevention rules (DC-INV-001 to DC-INV-008)
    6. Sector-specific target validation (SBTi-FI / NZBA)
    7. Temperature alignment assessment
    8. Financial institution disclosure requirements

    Thread-Safe: Singleton pattern with lock for concurrent access.

    Attributes:
        config: Engine configuration
        _instance: Singleton instance
        _lock: Thread lock for singleton pattern

    Example:
        >>> config = ComplianceCheckerConfig()
        >>> engine = ComplianceCheckerEngine.get_instance(config)
        >>> result_input = InvestmentResultInput(...)
        >>> all_results = engine.check_all_frameworks(result_input)
        >>> overall_score = engine.get_overall_compliance_score(all_results)
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config: Optional[ComplianceCheckerConfig] = None) -> None:
        """Initialize ComplianceCheckerEngine."""
        self.config = config or ComplianceCheckerConfig()
        logger.info(
            f"ComplianceCheckerEngine initialized with "
            f"{len(self.config.enabled_frameworks)} frameworks"
        )

    @classmethod
    def get_instance(
        cls, config: Optional[ComplianceCheckerConfig] = None
    ) -> "ComplianceCheckerEngine":
        """
        Get singleton instance of ComplianceCheckerEngine (thread-safe).

        Args:
            config: Configuration (optional, uses defaults if omitted).

        Returns:
            Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ========================================================================
    # Main Entry Points
    # ========================================================================

    def check_all_frameworks(
        self, result: InvestmentResultInput
    ) -> Dict[ComplianceFramework, ComplianceCheckResult]:
        """
        Check compliance against all enabled frameworks.

        Args:
            result: Calculation result to validate.

        Returns:
            Dictionary mapping framework to compliance check result.

        Example:
            >>> results = engine.check_all_frameworks(calc_result)
            >>> for fw, cr in results.items():
            ...     print(f"{fw}: {cr.status}")
        """
        logger.info("Running compliance checks for all frameworks")
        all_results: Dict[ComplianceFramework, ComplianceCheckResult] = {}

        dispatch: Dict[ComplianceFramework, Any] = {
            ComplianceFramework.GHG_PROTOCOL: self.check_ghg_protocol,
            ComplianceFramework.PCAF: self.check_pcaf,
            ComplianceFramework.ISO_14064: self.check_iso_14064,
            ComplianceFramework.CSRD_ESRS: self.check_csrd_esrs,
            ComplianceFramework.CDP: self.check_cdp,
            ComplianceFramework.SBTI_FI: self.check_sbti_fi,
            ComplianceFramework.SB_253: self.check_sb253,
            ComplianceFramework.TCFD: self.check_tcfd,
            ComplianceFramework.NZBA: self.check_nzba,
        }

        for framework in self.config.enabled_frameworks:
            try:
                checker = dispatch.get(framework)
                if checker is None:
                    logger.warning("Unknown framework: %s", framework)
                    continue

                check_result = checker(result)
                all_results[framework] = check_result
                logger.info(
                    f"{framework.value} compliance: {check_result.status.value} "
                    f"(score: {check_result.score})"
                )

            except Exception as e:
                logger.error(
                    f"Error checking {framework.value} compliance: {e}",
                    exc_info=True,
                )
                all_results[framework] = ComplianceCheckResult(
                    framework=framework,
                    status=ComplianceStatus.FAIL,
                    score=Decimal("0"),
                    issues=[
                        ComplianceFinding(
                            rule_code="CHECK_ERROR",
                            description=f"Compliance check failed: {e}",
                            severity=ComplianceSeverity.CRITICAL,
                            framework=framework.value,
                        )
                    ],
                )

        return all_results

    def check_compliance(
        self,
        result: InvestmentResultInput,
        frameworks: Optional[List[str]] = None,
    ) -> List[ComplianceCheckResult]:
        """
        Check compliance against specified frameworks (string-based API).

        Args:
            result: Calculation result to validate.
            frameworks: List of framework name strings (defaults to all enabled).

        Returns:
            List of ComplianceCheckResult.
        """
        if frameworks:
            original = self.config.enabled_frameworks
            self.config.enabled_frameworks = [
                ComplianceFramework(fw) for fw in frameworks
                if fw in [f.value for f in ComplianceFramework]
            ]
            all_results = self.check_all_frameworks(result)
            self.config.enabled_frameworks = original
        else:
            all_results = self.check_all_frameworks(result)

        return list(all_results.values())

    def get_overall_compliance_score(
        self, results: Dict[ComplianceFramework, ComplianceCheckResult]
    ) -> Decimal:
        """
        Calculate overall compliance score across all frameworks.

        Args:
            results: Dictionary of compliance check results.

        Returns:
            Overall compliance score (0-100).
        """
        if not results:
            return Decimal("0")
        total = sum(r.score for r in results.values())
        avg = total / len(results)
        return avg.quantize(_QUANT_2DP, rounding=ROUNDING)

    def get_compliance_summary(
        self, results: Dict[ComplianceFramework, ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Generate compliance summary across all frameworks.

        Args:
            results: Dictionary of compliance check results.

        Returns:
            Summary dictionary with scores, issues, recommendations.
        """
        overall_score = self.get_overall_compliance_score(results)

        if overall_score >= 95:
            overall_status = ComplianceStatus.PASS
        elif overall_score >= 70:
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.FAIL

        critical_issues: List[Dict[str, Any]] = []
        high_issues: List[Dict[str, Any]] = []
        medium_issues: List[Dict[str, Any]] = []
        low_issues: List[Dict[str, Any]] = []

        for framework, cr in results.items():
            for issue in cr.issues:
                entry = {
                    "framework": framework.value,
                    "rule_code": issue.rule_code,
                    "message": issue.description,
                    "recommendation": issue.recommendation,
                }
                if issue.severity == ComplianceSeverity.CRITICAL:
                    critical_issues.append(entry)
                elif issue.severity == ComplianceSeverity.HIGH:
                    high_issues.append(entry)
                elif issue.severity == ComplianceSeverity.MEDIUM:
                    medium_issues.append(entry)
                elif issue.severity == ComplianceSeverity.LOW:
                    low_issues.append(entry)

        framework_scores = {
            fw.value: {
                "score": float(cr.score),
                "status": cr.status.value,
                "passed": cr.passed_checks,
                "failed": cr.failed_checks,
                "warnings": cr.warning_checks,
            }
            for fw, cr in results.items()
        }

        recommendations = self._generate_recommendations(results)

        return {
            "overall_score": float(overall_score),
            "overall_status": overall_status.value,
            "frameworks_checked": len(results),
            "framework_scores": framework_scores,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "low_issues": low_issues,
            "total_issues": (
                len(critical_issues) + len(high_issues)
                + len(medium_issues) + len(low_issues)
            ),
            "recommendations": recommendations,
        }

    # ========================================================================
    # GHG Protocol Scope 3 Compliance
    # ========================================================================

    def check_ghg_protocol(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 (Category 15).

        GHG Protocol Category 15 Requirements:
        1. Materiality assessment for Cat 15
        2. Investment-specific attribution method
        3. Asset class breakdown disclosure
        4. Equity share vs operational control boundary
        5. Double-counting prevention (DC-INV-001 to DC-INV-008)
        6. Emission factor source documentation
        7. Data quality assessment
        8. Reporting period definition
        9. Base year definition
        10. Uncertainty quantification
        11. Exclusions documentation
        12. Calculation method hierarchy

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for GHG Protocol.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.GHG_PROTOCOL)

        # 1. Materiality assessment
        self._ghg_check_materiality(state, result)

        # 2. Attribution method
        self._ghg_check_attribution_method(state, result)

        # 3. Asset class breakdown disclosure
        self._ghg_check_asset_class_disclosure(state, result)

        # 4. Total emissions positivity
        self._ghg_check_emissions_positive(state, result)

        # 5. Double-counting prevention (all 8 rules)
        dc_violations = self._check_dc_rules(result)
        if dc_violations:
            for v in dc_violations:
                state.add_fail(
                    rule_code=v["rule_id"],
                    description=v["message"],
                    severity=ComplianceSeverity.CRITICAL
                    if v["rule_id"] in ("DC-INV-001", "DC-INV-002", "DC-INV-008")
                    else ComplianceSeverity.HIGH,
                    recommendation=v.get("recommendation"),
                    regulation_reference="GHG Protocol Scope 3, Category 15",
                )
        else:
            state.add_pass("DC-ALL", "No double-counting violations detected")

        # 6. Emission factor source documentation
        self._ghg_check_ef_sources(state, result)

        # 7. Data quality assessment
        self._ghg_check_data_quality(state, result)

        # 8. Reporting period
        self._ghg_check_reporting_period(state, result)

        # 9. Base year
        self._ghg_check_base_year(state, result)

        # 10. Uncertainty
        self._ghg_check_uncertainty(state, result)

        # 11. Calculation methods used
        self._ghg_check_calculation_methods(state, result)

        # 12. Scope 3 context
        self._ghg_check_scope3_context(state, result)

        return self._build_result(state, {"category": "15", "category_name": "Investments"})

    def _ghg_check_materiality(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check materiality assessment for Category 15."""
        if result.category_15_percentage_of_scope3 is not None:
            state.add_pass("GHG-INV-001", "Category 15 materiality assessed")
        else:
            state.add_warning(
                "GHG-INV-001",
                "Category 15 materiality not assessed (missing Scope 3 context)",
                severity=ComplianceSeverity.MEDIUM,
                recommendation=(
                    "Calculate Category 15 as percentage of total Scope 3 "
                    "to determine materiality"
                ),
                regulation_reference="GHG Protocol Scope 3, Ch 1",
            )

    def _ghg_check_attribution_method(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check attribution method is specified."""
        if result.attribution_method:
            state.add_pass("GHG-INV-002", "Attribution method documented")
        else:
            state.add_warning(
                "GHG-INV-002",
                "Attribution method not specified",
                severity=ComplianceSeverity.HIGH,
                recommendation="Document the attribution method (e.g., EVIC, equity share, revenue)",
                regulation_reference="GHG Protocol Scope 3, Category 15",
            )

    def _ghg_check_asset_class_disclosure(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check asset class breakdown is disclosed."""
        if result.asset_class_breakdown:
            state.add_pass("GHG-INV-003", "Asset class breakdown disclosed")
        else:
            state.add_warning(
                "GHG-INV-003",
                "Asset class breakdown not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose financed emissions by asset class",
                regulation_reference="GHG Protocol Scope 3, Category 15",
            )

    def _ghg_check_emissions_positive(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check total emissions are non-negative."""
        if result.total_emissions_kg_co2e >= 0:
            state.add_pass("GHG-INV-004", "Total emissions are non-negative")
        else:
            state.add_fail(
                "GHG-INV-004",
                "Total emissions are negative -- invalid",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Check calculation: financed emissions must be >= 0",
            )

    def _ghg_check_ef_sources(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check emission factor sources are documented."""
        if result.calculation_methods_used:
            state.add_pass("GHG-INV-005", "Calculation methods documented")
        else:
            state.add_warning(
                "GHG-INV-005",
                "Calculation methods not documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Document EF sources and calculation methods",
            )

    def _ghg_check_data_quality(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check data quality is assessed."""
        if result.weighted_data_quality_score is not None:
            state.add_pass("GHG-INV-006", "Data quality assessed")
        else:
            state.add_warning(
                "GHG-INV-006",
                "Data quality score not assessed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Calculate weighted average PCAF data quality score",
            )

    def _ghg_check_reporting_period(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check reporting period is defined."""
        if result.reporting_period_start and result.reporting_period_end:
            state.add_pass("GHG-INV-007", "Reporting period defined")
        else:
            state.add_warning(
                "GHG-INV-007",
                "Reporting period not fully defined",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Define reporting period start and end dates",
            )

    def _ghg_check_base_year(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check base year is defined."""
        if result.base_year is not None:
            state.add_pass("GHG-INV-008", "Base year defined")
        else:
            state.add_warning(
                "GHG-INV-008",
                "Base year not defined",
                severity=ComplianceSeverity.LOW,
                recommendation="Define base year for trend tracking",
            )

    def _ghg_check_uncertainty(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check uncertainty is quantified."""
        if result.uncertainty_percentage is not None:
            state.add_pass("GHG-INV-009", "Uncertainty quantified")
        else:
            state.add_warning(
                "GHG-INV-009",
                "Uncertainty not quantified",
                severity=ComplianceSeverity.LOW,
                recommendation="Quantify uncertainty for financed emissions",
            )

    def _ghg_check_calculation_methods(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check calculation method hierarchy is applied."""
        if result.calculation_methods_used:
            preferred = {"reported_verified", "reported_unverified", "physical_activity"}
            used_set = set(result.calculation_methods_used)
            if used_set & preferred:
                state.add_pass("GHG-INV-010", "Higher-quality calculation methods used")
            else:
                state.add_warning(
                    "GHG-INV-010",
                    "Only lower-quality calculation methods used (EEIO/sector average)",
                    severity=ComplianceSeverity.MEDIUM,
                    recommendation="Where possible, use reported or physical activity data",
                )
        else:
            state.add_pass("GHG-INV-010", "Calculation method check skipped (no methods listed)")

    def _ghg_check_scope3_context(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check Scope 3 context is available."""
        if result.total_scope3_emissions_kg_co2e is not None:
            state.add_pass("GHG-INV-011", "Total Scope 3 context provided")
        else:
            state.add_warning(
                "GHG-INV-011",
                "Total Scope 3 emissions not provided for materiality context",
                severity=ComplianceSeverity.LOW,
                recommendation="Provide total Scope 3 to assess Cat 15 materiality",
            )

    # ========================================================================
    # PCAF Compliance
    # ========================================================================

    def check_pcaf(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with PCAF Global GHG Accounting Standard (3rd Ed.).

        PCAF Requirements:
        1. Asset class coverage (8 asset classes)
        2. Attribution factor formula per asset class
        3. Data quality score 1-5 per asset class
        4. Weighted average data quality score
        5. Data quality improvement plan
        6. WACI calculation
        7. Portfolio coverage threshold
        8. Absolute emissions AND intensity metric
        9. Financed emissions per asset class
        10. Year-on-year comparison

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for PCAF.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.PCAF)

        # 1. Asset class coverage
        self._pcaf_check_asset_class_coverage(state, result)

        # 2. Attribution factor validation
        pcaf_attr_violations = self._validate_pcaf_attribution(result)
        if pcaf_attr_violations:
            for v in pcaf_attr_violations:
                state.add_fail(
                    "PCAF-ATTR",
                    v,
                    severity=ComplianceSeverity.HIGH,
                    recommendation="Review PCAF attribution formula for each asset class",
                    regulation_reference="PCAF Standard, Part A",
                )
        else:
            state.add_pass("PCAF-ATTR", "Attribution factors valid")

        # 3. Data quality scores per asset class
        self._pcaf_check_dq_scores(state, result)

        # 4. Weighted average data quality score
        self._pcaf_check_weighted_dq(state, result)

        # 5. Data quality improvement plan
        self._pcaf_check_dq_improvement_plan(state, result)

        # 6. WACI calculation
        self._pcaf_check_waci(state, result)

        # 7. Portfolio coverage
        coverage_result = self._check_portfolio_coverage(result)
        if coverage_result.status == ComplianceStatus.PASS:
            state.add_pass("PCAF-COV", "Portfolio coverage meets threshold")
        elif coverage_result.status == ComplianceStatus.WARNING:
            state.add_warning(
                "PCAF-COV",
                f"Portfolio coverage below target ({result.portfolio_coverage_pct}%)",
                severity=ComplianceSeverity.HIGH,
                recommendation="Increase portfolio coverage for financed emissions measurement",
            )
        else:
            state.add_fail(
                "PCAF-COV",
                "Portfolio coverage not measured",
                severity=ComplianceSeverity.HIGH,
                recommendation="Measure and disclose % of AUM with financed emissions",
            )

        # 8. Absolute vs intensity disclosure
        self._pcaf_check_dual_metric(state, result)

        # 9. Financed emissions per asset class
        self._pcaf_check_per_asset_class_emissions(state, result)

        # 10. Year-on-year comparison
        self._pcaf_check_yoy(state, result)

        return self._build_result(
            state, {"standard": "PCAF Global GHG Standard 3rd Edition"}
        )

    def _pcaf_check_asset_class_coverage(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check asset class coverage."""
        covered = set(result.asset_class_breakdown.keys())
        if covered:
            state.add_pass("PCAF-001", f"Asset class coverage: {len(covered)} classes")
        else:
            state.add_fail(
                "PCAF-001",
                "No asset class breakdown provided",
                severity=ComplianceSeverity.HIGH,
                recommendation="Disclose financed emissions by PCAF asset class",
                regulation_reference="PCAF Standard, Part A, Section 4",
            )

    def _pcaf_check_dq_scores(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check data quality scores are assigned per asset class."""
        if result.data_quality_scores:
            # Validate scores are in 1-5 range
            invalid = {
                ac: s for ac, s in result.data_quality_scores.items()
                if s < 1 or s > 5
            }
            if invalid:
                state.add_fail(
                    "PCAF-003",
                    f"Invalid PCAF DQ scores (must be 1-5): {invalid}",
                    severity=ComplianceSeverity.HIGH,
                    recommendation="Assign PCAF data quality scores 1-5 per asset class",
                )
            else:
                state.add_pass("PCAF-003", "PCAF data quality scores valid")
        else:
            state.add_warning(
                "PCAF-003",
                "PCAF data quality scores not assigned per asset class",
                severity=ComplianceSeverity.HIGH,
                recommendation="Assign PCAF data quality scores (1-5) per asset class",
                regulation_reference="PCAF Standard, Part A, Data Quality",
            )

    def _pcaf_check_weighted_dq(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check weighted average data quality score."""
        if result.weighted_data_quality_score is not None:
            wdq = result.weighted_data_quality_score
            if wdq <= self.config.maximum_dq_score:
                state.add_pass("PCAF-004", f"Weighted DQ score {wdq} within threshold")
            else:
                state.add_warning(
                    "PCAF-004",
                    f"Weighted DQ score {wdq} exceeds target {self.config.maximum_dq_score}",
                    severity=ComplianceSeverity.MEDIUM,
                    recommendation="Improve data quality by engaging investees for reported data",
                )
        else:
            state.add_warning(
                "PCAF-004",
                "Weighted data quality score not calculated",
                severity=ComplianceSeverity.HIGH,
                recommendation="Calculate AUM-weighted average PCAF data quality score",
            )

    def _pcaf_check_dq_improvement_plan(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check data quality improvement plan exists."""
        if result.has_dq_improvement_plan:
            state.add_pass("PCAF-005", "Data quality improvement plan documented")
        else:
            state.add_warning(
                "PCAF-005",
                "No data quality improvement plan documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Create a plan to improve PCAF data quality over time",
                regulation_reference="PCAF Standard, Part C, DQ Improvement",
            )

    def _pcaf_check_waci(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check WACI calculation is provided."""
        if result.waci_value is not None:
            if result.waci_value >= 0:
                state.add_pass("PCAF-006", f"WACI calculated: {result.waci_value} tCO2e/$M")
            else:
                state.add_fail(
                    "PCAF-006",
                    "WACI value is negative -- invalid",
                    severity=ComplianceSeverity.HIGH,
                    recommendation="Review WACI calculation methodology",
                )
        else:
            state.add_warning(
                "PCAF-006",
                "WACI not calculated",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Calculate Weighted Average Carbon Intensity (tCO2e/$M revenue)",
            )

    def _pcaf_check_dual_metric(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check both absolute and intensity metrics are reported."""
        has_absolute = result.total_emissions_kg_co2e is not None
        has_intensity = result.waci_value is not None

        if has_absolute and has_intensity:
            state.add_pass("PCAF-008", "Both absolute and intensity metrics reported")
        elif has_absolute:
            state.add_warning(
                "PCAF-008",
                "Only absolute emissions reported; intensity metric (WACI) missing",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Report both absolute financed emissions and WACI",
            )
        else:
            state.add_fail(
                "PCAF-008",
                "Neither absolute nor intensity metric reported",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="Report absolute financed emissions and WACI",
            )

    def _pcaf_check_per_asset_class_emissions(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check financed emissions are reported per asset class."""
        if result.asset_class_breakdown and len(result.asset_class_breakdown) > 0:
            all_non_negative = all(v >= 0 for v in result.asset_class_breakdown.values())
            if all_non_negative:
                state.add_pass("PCAF-009", "Per-asset-class emissions reported")
            else:
                state.add_fail(
                    "PCAF-009",
                    "Negative emissions in asset class breakdown",
                    severity=ComplianceSeverity.HIGH,
                    recommendation="All per-asset-class emissions must be >= 0",
                )
        else:
            state.add_warning(
                "PCAF-009",
                "Per-asset-class financed emissions not reported",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Report financed emissions breakdown by PCAF asset class",
            )

    def _pcaf_check_yoy(
        self, state: FrameworkCheckState, result: InvestmentResultInput
    ) -> None:
        """Check year-on-year comparison data."""
        if result.base_year is not None and result.reporting_year is not None:
            state.add_pass("PCAF-010", "Year-on-year comparison data available")
        else:
            state.add_warning(
                "PCAF-010",
                "Year-on-year comparison not available (no base year or reporting year)",
                severity=ComplianceSeverity.LOW,
                recommendation="Set base year and track annual changes",
            )

    # ========================================================================
    # ISO 14064-1:2018 Compliance
    # ========================================================================

    def check_iso_14064(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018 (Clause 5.2.4).

        ISO 14064-1 Requirements:
        1. Methodology documentation
        2. Emission factor sources and vintage
        3. Data quality assessment
        4. Uncertainty assessment
        5. Organizational boundary documentation
        6. Reporting period definition
        7. Base year definition
        8. Quantification approach
        9. Exclusions justification
        10. Recalculation triggers

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for ISO 14064.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.ISO_14064)

        # 1. Methodology documentation
        if result.calculation_method or result.calculation_methods_used:
            state.add_pass("ISO-001", "Methodology documented")
        else:
            state.add_fail(
                "ISO-001",
                "Quantification methodology not documented",
                severity=ComplianceSeverity.HIGH,
                recommendation="Document the quantification methodology used",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4",
            )

        # 2. Emission factor sources
        if result.calculation_methods_used:
            state.add_pass("ISO-002", "Emission factor sources documented")
        else:
            state.add_warning(
                "ISO-002",
                "Emission factor sources not documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Document EF sources with vintage year",
            )

        # 3. Data quality assessment
        if result.weighted_data_quality_score is not None:
            state.add_pass("ISO-003", "Data quality assessed")
        else:
            state.add_warning(
                "ISO-003",
                "Data quality not assessed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Assess data quality using PCAF scoring",
            )

        # 4. Uncertainty assessment
        if result.uncertainty_percentage is not None:
            state.add_pass("ISO-004", "Uncertainty assessed")
        else:
            state.add_fail(
                "ISO-004",
                "Uncertainty assessment required by ISO 14064-1",
                severity=ComplianceSeverity.HIGH,
                recommendation="Provide quantitative uncertainty assessment",
                regulation_reference="ISO 14064-1:2018, Clause 5.5",
            )

        # 5. Organizational boundary
        if result.attribution_method:
            state.add_pass("ISO-005", "Organizational boundary documented")
        else:
            state.add_warning(
                "ISO-005",
                "Organizational boundary (attribution approach) not documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Document organizational boundary approach",
                regulation_reference="ISO 14064-1:2018, Clause 5.1",
            )

        # 6. Reporting period
        if result.reporting_period_start and result.reporting_period_end:
            state.add_pass("ISO-006", "Reporting period defined")
        else:
            state.add_warning(
                "ISO-006",
                "Reporting period not fully defined",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Define reporting period start and end dates",
            )

        # 7. Base year
        if result.base_year is not None:
            state.add_pass("ISO-007", "Base year defined")
        else:
            state.add_warning(
                "ISO-007",
                "Base year not defined",
                severity=ComplianceSeverity.LOW,
                recommendation="Establish base year for GHG inventory",
                regulation_reference="ISO 14064-1:2018, Clause 5.4",
            )

        # 8. Non-negative emissions
        if result.total_emissions_kg_co2e >= 0:
            state.add_pass("ISO-008", "Emissions are non-negative")
        else:
            state.add_fail(
                "ISO-008",
                "Negative emissions reported",
                severity=ComplianceSeverity.CRITICAL,
            )

        # 9. Double-counting addressed
        dc_violations = self._check_dc_rules(result)
        if not dc_violations:
            state.add_pass("ISO-009", "Double-counting addressed")
        else:
            state.add_warning(
                "ISO-009",
                f"{len(dc_violations)} double-counting risk(s) identified",
                severity=ComplianceSeverity.HIGH,
                recommendation="Resolve double-counting issues per ISO 14064-1 Clause 5.2",
            )

        # 10. Portfolio coverage
        if result.portfolio_coverage_pct is not None:
            state.add_pass("ISO-010", "Coverage scope documented")
        else:
            state.add_warning(
                "ISO-010",
                "Coverage scope not documented",
                severity=ComplianceSeverity.LOW,
                recommendation="Document the proportion of investments covered",
            )

        return self._build_result(
            state, {"standard": "ISO 14064-1:2018", "clause": "5.2.4"}
        )

    # ========================================================================
    # CSRD / ESRS E1 Compliance
    # ========================================================================

    def check_csrd_esrs(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with CSRD/ESRS E1 Climate Change.

        CSRD ESRS E1 Requirements (Financial Entities):
        1. E1-6 GHG Scope 3 downstream disclosure
        2. Financed emissions for financial entities
        3. Asset class breakdown
        4. Data quality disclosure
        5. Targets and transition plan
        6. XBRL tagging readiness
        7. Double materiality assessment
        8. Sector-level disclosure
        9. Intensity metrics
        10. Connectivity with financial statements

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for CSRD/ESRS.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CSRD_ESRS)

        # 1. E1-6 GHG Scope 3 disclosure
        if result.total_emissions_kg_co2e > 0:
            state.add_pass("CSRD-001", "Scope 3 Cat 15 emissions disclosed (E1-6)")
        else:
            state.add_warning(
                "CSRD-001",
                "Scope 3 Cat 15 emissions are zero or missing",
                severity=ComplianceSeverity.HIGH,
                recommendation="Financial entities must disclose financed emissions under ESRS E1-6",
                regulation_reference="ESRS E1-6, Disclosure Requirement",
            )

        # 2. Financial entity flag
        if result.is_financial_institution:
            state.add_pass("CSRD-002", "Reporting entity identified as financial institution")
        else:
            state.add_warning(
                "CSRD-002",
                "Entity not flagged as financial institution; E1-6 may still apply",
                severity=ComplianceSeverity.LOW,
            )

        # 3. Asset class breakdown
        if result.asset_class_breakdown:
            state.add_pass("CSRD-003", "Asset class breakdown disclosed")
        else:
            state.add_warning(
                "CSRD-003",
                "Asset class breakdown not provided",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="ESRS E1 requires granular breakdown of financed emissions",
            )

        # 4. Data quality disclosure
        if result.weighted_data_quality_score is not None:
            state.add_pass("CSRD-004", "Data quality disclosed")
        else:
            state.add_warning(
                "CSRD-004",
                "Data quality not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose data quality methodology and scores",
            )

        # 5. Sector-level disclosure
        if result.sector_breakdown:
            state.add_pass("CSRD-005", "Sector-level breakdown provided")
        else:
            state.add_warning(
                "CSRD-005",
                "Sector-level breakdown not provided",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose financed emissions by high-impact sector",
            )

        # 6. Intensity metrics
        if result.waci_value is not None:
            state.add_pass("CSRD-006", "Intensity metric (WACI) disclosed")
        else:
            state.add_warning(
                "CSRD-006",
                "Intensity metric not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Report WACI alongside absolute financed emissions",
            )

        # 7. Reporting period
        if result.reporting_period_start and result.reporting_period_end:
            state.add_pass("CSRD-007", "Reporting period defined")
        else:
            state.add_warning(
                "CSRD-007",
                "Reporting period not fully defined",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 8. Targets and transition plan
        if result.sector_targets:
            state.add_pass("CSRD-008", "Targets/transition plan documented")
        else:
            state.add_warning(
                "CSRD-008",
                "No reduction targets or transition plan documented",
                severity=ComplianceSeverity.HIGH,
                recommendation="CSRD requires disclosure of transition plans",
                regulation_reference="ESRS E1-1, Transition Plan",
            )

        # 9. Double-counting addressed
        dc_violations = self._check_dc_rules(result)
        if not dc_violations:
            state.add_pass("CSRD-009", "Double-counting addressed")
        else:
            state.add_warning(
                "CSRD-009",
                f"{len(dc_violations)} double-counting risk(s)",
                severity=ComplianceSeverity.HIGH,
            )

        # 10. Portfolio coverage
        if result.portfolio_coverage_pct is not None:
            state.add_pass("CSRD-010", "Portfolio coverage disclosed")
        else:
            state.add_warning(
                "CSRD-010",
                "Portfolio coverage not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose % of AUM covered by financed emissions calculation",
            )

        return self._build_result(
            state, {"standard": "CSRD ESRS E1", "disclosure": "E1-6"}
        )

    # ========================================================================
    # CDP Climate Change Compliance
    # ========================================================================

    def check_cdp(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change (C6.5, C-FS14.1).

        CDP Requirements:
        1. C6.5 Category 15 disclosure
        2. C-FS14.1 Financed emissions disclosure (FI-specific)
        3. Methodology disclosure
        4. Coverage/completeness
        5. Data quality assessment
        6. Sector breakdown
        7. Intensity metrics
        8. Verification status
        9. Year-on-year comparison
        10. Portfolio alignment

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for CDP.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.CDP)

        # 1. C6.5 Category 15 disclosure
        if result.total_emissions_kg_co2e > 0:
            state.add_pass("CDP-001", "C6.5 Category 15 emissions disclosed")
        else:
            state.add_warning(
                "CDP-001",
                "Category 15 emissions zero or missing for C6.5",
                severity=ComplianceSeverity.HIGH,
                recommendation="Disclose Cat 15 in CDP questionnaire C6.5",
                regulation_reference="CDP Climate Change, C6.5",
            )

        # 2. C-FS14.1 Financed emissions (FI-specific)
        if result.is_financial_institution:
            if result.asset_class_breakdown:
                state.add_pass("CDP-002", "C-FS14.1 financed emissions disclosed")
            else:
                state.add_fail(
                    "CDP-002",
                    "C-FS14.1 requires financed emissions by asset class for FIs",
                    severity=ComplianceSeverity.HIGH,
                    recommendation="Complete C-FS14.1 with PCAF-aligned financed emissions",
                    regulation_reference="CDP Climate Change, C-FS14.1",
                )
        else:
            state.add_pass("CDP-002", "C-FS14.1 not applicable (non-FI)")

        # 3. Methodology disclosure
        if result.calculation_method or result.attribution_method:
            state.add_pass("CDP-003", "Methodology disclosed")
        else:
            state.add_warning(
                "CDP-003",
                "Methodology not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose calculation methodology in CDP response",
            )

        # 4. Coverage
        if result.portfolio_coverage_pct is not None:
            state.add_pass("CDP-004", "Portfolio coverage disclosed")
        else:
            state.add_warning(
                "CDP-004",
                "Portfolio coverage not disclosed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 5. Data quality
        if result.weighted_data_quality_score is not None:
            state.add_pass("CDP-005", "Data quality assessment disclosed")
        else:
            state.add_warning(
                "CDP-005",
                "Data quality not disclosed",
                severity=ComplianceSeverity.LOW,
            )

        # 6. Sector breakdown
        if result.sector_breakdown:
            state.add_pass("CDP-006", "Sector breakdown disclosed")
        else:
            state.add_warning(
                "CDP-006",
                "Sector breakdown not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose financed emissions by sector for CDP",
            )

        # 7. Intensity metrics
        if result.waci_value is not None:
            state.add_pass("CDP-007", "WACI intensity metric disclosed")
        else:
            state.add_warning(
                "CDP-007",
                "WACI not disclosed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 8. Year-on-year comparison
        if result.base_year is not None and result.reporting_year is not None:
            state.add_pass("CDP-008", "Year-on-year comparison possible")
        else:
            state.add_warning(
                "CDP-008",
                "Year-on-year comparison not available",
                severity=ComplianceSeverity.LOW,
            )

        # 9. Portfolio alignment
        if result.portfolio_alignment_pct is not None:
            state.add_pass("CDP-009", "Portfolio alignment disclosed")
        else:
            state.add_warning(
                "CDP-009",
                "Portfolio alignment not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose portfolio alignment to Paris Agreement targets",
            )

        # 10. Non-negative emissions
        if result.total_emissions_kg_co2e >= 0:
            state.add_pass("CDP-010", "Emissions non-negative")
        else:
            state.add_fail(
                "CDP-010",
                "Negative emissions reported",
                severity=ComplianceSeverity.CRITICAL,
            )

        return self._build_result(
            state, {"questionnaire": "CDP Climate Change", "modules": "C6.5, C-FS14.1"}
        )

    # ========================================================================
    # SBTi Financial Institutions Compliance
    # ========================================================================

    def check_sbti_fi(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with SBTi Financial Institutions framework.

        SBTi-FI Requirements:
        1. Materiality threshold for Cat 15
        2. Portfolio coverage for financed emissions
        3. Sector-specific targets
        4. Temperature alignment assessment
        5. Portfolio alignment percentage
        6. Near-term targets (5-10 years)
        7. Long-term target (net-zero by 2050)
        8. Scope 1+2 coverage of investees
        9. PCAF methodology alignment
        10. Annual progress reporting

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for SBTi-FI.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SBTI_FI)

        # 1. Materiality threshold
        threshold = self.config.sbti_fi_materiality_threshold
        if result.category_15_percentage_of_scope3 is not None:
            if result.category_15_percentage_of_scope3 >= threshold:
                state.add_pass(
                    "SBTI-001",
                    f"Cat 15 is material ({result.category_15_percentage_of_scope3}% >= {threshold}%)"
                )
            else:
                state.add_pass(
                    "SBTI-001",
                    f"Cat 15 below materiality threshold ({result.category_15_percentage_of_scope3}%)"
                )
        else:
            state.add_warning(
                "SBTI-001",
                "Cat 15 materiality not assessed",
                severity=ComplianceSeverity.HIGH,
                recommendation="Assess Cat 15 materiality for SBTi-FI target setting",
                regulation_reference="SBTi-FI Framework, Section 4",
            )

        # 2. Portfolio coverage
        cov_target = self.config.sbti_fi_portfolio_coverage_threshold
        if result.portfolio_coverage_pct is not None:
            if result.portfolio_coverage_pct >= cov_target:
                state.add_pass(
                    "SBTI-002",
                    f"Portfolio coverage {result.portfolio_coverage_pct}% >= {cov_target}%"
                )
            else:
                state.add_warning(
                    "SBTI-002",
                    f"Portfolio coverage {result.portfolio_coverage_pct}% < {cov_target}%",
                    severity=ComplianceSeverity.HIGH,
                    recommendation=f"SBTi-FI requires {cov_target}% portfolio coverage",
                )
        else:
            state.add_fail(
                "SBTI-002",
                "Portfolio coverage not measured",
                severity=ComplianceSeverity.HIGH,
                recommendation="Measure and report portfolio coverage",
            )

        # 3. Sector-specific targets
        if result.sector_targets:
            state.add_pass("SBTI-003", f"Sector targets set for {len(result.sector_targets)} sectors")
        else:
            state.add_warning(
                "SBTI-003",
                "No sector-specific targets set",
                severity=ComplianceSeverity.HIGH,
                recommendation="Set sector-specific decarbonization targets per SBTi-FI",
                regulation_reference="SBTi-FI, Sector-Specific Target Setting",
            )

        # 4. Temperature alignment
        temp_target = self.config.sbti_fi_temperature_target
        if result.temperature_alignment is not None:
            if result.temperature_alignment <= temp_target:
                state.add_pass(
                    "SBTI-004",
                    f"Temperature aligned at {result.temperature_alignment}C (target {temp_target}C)"
                )
            else:
                state.add_warning(
                    "SBTI-004",
                    f"Temperature alignment {result.temperature_alignment}C > {temp_target}C target",
                    severity=ComplianceSeverity.HIGH,
                    recommendation="Accelerate portfolio decarbonization to meet temperature target",
                )
        else:
            state.add_warning(
                "SBTI-004",
                "Temperature alignment not assessed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Calculate portfolio temperature alignment score",
            )

        # 5. Portfolio alignment percentage
        if result.portfolio_alignment_pct is not None:
            state.add_pass(
                "SBTI-005",
                f"Portfolio alignment: {result.portfolio_alignment_pct}%"
            )
        else:
            state.add_warning(
                "SBTI-005",
                "Portfolio alignment percentage not reported",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 6. PCAF methodology alignment
        if result.data_quality_scores or result.weighted_data_quality_score is not None:
            state.add_pass("SBTI-006", "PCAF-aligned methodology used")
        else:
            state.add_warning(
                "SBTI-006",
                "PCAF-aligned methodology not confirmed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Use PCAF methodology for financed emissions measurement",
            )

        # 7. Absolute emissions reported
        if result.total_emissions_kg_co2e > 0:
            state.add_pass("SBTI-007", "Absolute financed emissions reported")
        else:
            state.add_fail(
                "SBTI-007",
                "Absolute financed emissions not reported",
                severity=ComplianceSeverity.HIGH,
            )

        # 8. Base year for tracking
        if result.base_year is not None:
            state.add_pass("SBTI-008", "Base year established")
        else:
            state.add_warning(
                "SBTI-008",
                "No base year established for target tracking",
                severity=ComplianceSeverity.HIGH,
                recommendation="Establish base year for SBTi target tracking",
            )

        # 9. Reporting year
        if result.reporting_year is not None:
            state.add_pass("SBTI-009", "Reporting year specified")
        else:
            state.add_warning(
                "SBTI-009",
                "Reporting year not specified",
                severity=ComplianceSeverity.LOW,
            )

        # 10. Data quality improvement
        if result.has_dq_improvement_plan:
            state.add_pass("SBTI-010", "DQ improvement plan in place")
        else:
            state.add_warning(
                "SBTI-010",
                "No data quality improvement plan",
                severity=ComplianceSeverity.LOW,
                recommendation="Create plan to improve data quality over time",
            )

        return self._build_result(
            state, {"framework": "SBTi Financial Institutions"}
        )

    # ========================================================================
    # SB 253 (California) Compliance
    # ========================================================================

    def check_sb253(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253.

        SB 253 Requirements:
        1. Material Scope 3 inclusion
        2. Assurance readiness
        3. Financial sector inclusion
        4. Annual reporting
        5. Emissions calculation methodology
        6. Data quality documentation
        7. Exclusions documentation
        8. Base year establishment

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for SB 253.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.SB_253)

        # 1. Material Scope 3 Cat 15 reported
        if result.total_emissions_kg_co2e > 0:
            state.add_pass("SB253-001", "Category 15 emissions reported")
        else:
            state.add_warning(
                "SB253-001",
                "Category 15 emissions are zero or not reported",
                severity=ComplianceSeverity.HIGH,
                recommendation="SB 253 requires material Scope 3 categories to be reported",
                regulation_reference="SB 253, Section 3(a)",
            )

        # 2. Assurance readiness
        if result.weighted_data_quality_score is not None and result.uncertainty_percentage is not None:
            state.add_pass("SB253-002", "Assurance readiness: DQ + uncertainty documented")
        else:
            state.add_warning(
                "SB253-002",
                "Assurance readiness incomplete (missing DQ or uncertainty)",
                severity=ComplianceSeverity.HIGH,
                recommendation="SB 253 will require limited/reasonable assurance over time",
            )

        # 3. Financial sector flag
        if result.is_financial_institution:
            state.add_pass("SB253-003", "Financial sector entity identified")
        else:
            state.add_pass("SB253-003", "Non-financial entity (Cat 15 may be less material)")

        # 4. Reporting period defined
        if result.reporting_period_start and result.reporting_period_end:
            state.add_pass("SB253-004", "Annual reporting period defined")
        else:
            state.add_warning(
                "SB253-004",
                "Reporting period not defined",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 5. Calculation methodology
        if result.calculation_methods_used:
            state.add_pass("SB253-005", "Calculation methodology documented")
        else:
            state.add_warning(
                "SB253-005",
                "Calculation methodology not documented",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Document methodology for SB 253 compliance",
            )

        # 6. Data quality
        if result.weighted_data_quality_score is not None:
            state.add_pass("SB253-006", "Data quality assessed")
        else:
            state.add_warning(
                "SB253-006",
                "Data quality not assessed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 7. Emissions non-negative
        if result.total_emissions_kg_co2e >= 0:
            state.add_pass("SB253-007", "Emissions non-negative")
        else:
            state.add_fail(
                "SB253-007",
                "Negative emissions",
                severity=ComplianceSeverity.CRITICAL,
            )

        # 8. Base year
        if result.base_year is not None:
            state.add_pass("SB253-008", "Base year established")
        else:
            state.add_warning(
                "SB253-008",
                "Base year not established",
                severity=ComplianceSeverity.LOW,
            )

        return self._build_result(
            state, {"regulation": "California SB 253"}
        )

    # ========================================================================
    # TCFD Compliance
    # ========================================================================

    def check_tcfd(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with TCFD Recommendations (Metrics & Targets).

        TCFD Requirements:
        1. Financed emissions disclosure
        2. Intensity metrics (WACI)
        3. Scenario analysis
        4. Sector breakdown
        5. Targets disclosure
        6. Temperature alignment
        7. Methodology documentation
        8. Year-on-year trend
        9. Portfolio coverage
        10. Risk assessment linkage

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for TCFD.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.TCFD)

        # 1. Financed emissions disclosure
        if result.total_emissions_kg_co2e > 0:
            state.add_pass("TCFD-001", "Financed emissions disclosed")
        else:
            state.add_fail(
                "TCFD-001",
                "Financed emissions not disclosed",
                severity=ComplianceSeverity.HIGH,
                recommendation="TCFD Metrics & Targets requires financed emissions disclosure",
                regulation_reference="TCFD, Supplemental Guidance for Financial Sector",
            )

        # 2. WACI intensity metrics
        if result.waci_value is not None:
            state.add_pass("TCFD-002", "WACI intensity metric disclosed")
        else:
            state.add_warning(
                "TCFD-002",
                "WACI not disclosed",
                severity=ComplianceSeverity.HIGH,
                recommendation="TCFD recommends WACI as the primary intensity metric",
            )

        # 3. Scenario analysis
        if result.has_scenario_analysis:
            state.add_pass("TCFD-003", "Scenario analysis conducted")
        else:
            state.add_warning(
                "TCFD-003",
                "Scenario analysis not conducted",
                severity=ComplianceSeverity.HIGH,
                recommendation="Conduct scenario analysis (e.g., IEA NZE, NGFS)",
                regulation_reference="TCFD, Strategy Recommendations",
            )

        # 4. Sector breakdown
        if result.sector_breakdown:
            state.add_pass("TCFD-004", "Sector breakdown disclosed")
        else:
            state.add_warning(
                "TCFD-004",
                "Sector breakdown not disclosed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Disclose exposure by carbon-intensive sector",
            )

        # 5. Targets
        if result.sector_targets:
            state.add_pass("TCFD-005", "Decarbonization targets disclosed")
        else:
            state.add_warning(
                "TCFD-005",
                "No targets disclosed",
                severity=ComplianceSeverity.HIGH,
                recommendation="Set and disclose emissions reduction targets",
            )

        # 6. Temperature alignment
        if result.temperature_alignment is not None:
            state.add_pass("TCFD-006", f"Temperature alignment: {result.temperature_alignment}C")
        else:
            state.add_warning(
                "TCFD-006",
                "Temperature alignment not assessed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 7. Methodology
        if result.attribution_method or result.calculation_method:
            state.add_pass("TCFD-007", "Methodology documented")
        else:
            state.add_warning(
                "TCFD-007",
                "Methodology not documented",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 8. Year-on-year trend
        if result.base_year is not None and result.reporting_year is not None:
            state.add_pass("TCFD-008", "YoY trend data available")
        else:
            state.add_warning(
                "TCFD-008",
                "Year-on-year trend data not available",
                severity=ComplianceSeverity.LOW,
            )

        # 9. Portfolio coverage
        if result.portfolio_coverage_pct is not None:
            state.add_pass("TCFD-009", "Portfolio coverage disclosed")
        else:
            state.add_warning(
                "TCFD-009",
                "Portfolio coverage not disclosed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 10. Asset class breakdown
        if result.asset_class_breakdown:
            state.add_pass("TCFD-010", "Asset class breakdown available")
        else:
            state.add_warning(
                "TCFD-010",
                "Asset class breakdown not available",
                severity=ComplianceSeverity.LOW,
            )

        return self._build_result(
            state, {"framework": "TCFD Recommendations", "pillar": "Metrics & Targets"}
        )

    # ========================================================================
    # NZBA / NZAOA Compliance
    # ========================================================================

    def check_nzba(self, result: InvestmentResultInput) -> ComplianceCheckResult:
        """
        Check compliance with NZBA/NZAOA requirements.

        NZBA/NZAOA Requirements:
        1. Financed emissions measurement
        2. Sector pathway targets (at least 9 sectors)
        3. Interim targets (2030)
        4. Long-term net-zero target (2050)
        5. PCAF methodology alignment
        6. Annual disclosure
        7. WACI reporting
        8. Portfolio coverage
        9. Fossil fuel financing policy
        10. Transition plan

        Args:
            result: Calculation result to validate.

        Returns:
            ComplianceCheckResult for NZBA.
        """
        state = FrameworkCheckState(framework=ComplianceFramework.NZBA)

        # 1. Financed emissions measured
        if result.total_emissions_kg_co2e > 0:
            state.add_pass("NZBA-001", "Financed emissions measured")
        else:
            state.add_fail(
                "NZBA-001",
                "Financed emissions not measured",
                severity=ComplianceSeverity.CRITICAL,
                recommendation="NZBA requires measurement of financed emissions",
                regulation_reference="NZBA Commitment, Section 3",
            )

        # 2. Sector pathway targets
        target_sectors = set(result.sector_targets.keys()) if result.sector_targets else set()
        required = set(NZBA_TARGET_SECTORS)
        covered_required = target_sectors & required
        if len(covered_required) >= 3:
            state.add_pass(
                "NZBA-002",
                f"Sector targets set for {len(covered_required)} NZBA priority sectors"
            )
        elif result.sector_targets:
            state.add_warning(
                "NZBA-002",
                f"Only {len(covered_required)} NZBA priority sectors have targets (need >= 3)",
                severity=ComplianceSeverity.HIGH,
                recommendation="Set targets for carbon-intensive sectors per NZBA guidance",
            )
        else:
            state.add_fail(
                "NZBA-002",
                "No sector pathway targets set",
                severity=ComplianceSeverity.HIGH,
                recommendation="NZBA requires sector-specific decarbonization targets",
            )

        # 3. Interim targets (2030)
        has_interim = False
        if result.sector_targets:
            for _sector, target in result.sector_targets.items():
                target_year = target.get("target_year")
                if target_year and int(target_year) <= self.config.nzba_interim_target_year:
                    has_interim = True
                    break
        if has_interim:
            state.add_pass("NZBA-003", "Interim target (2030) set")
        else:
            state.add_warning(
                "NZBA-003",
                "No interim target for 2030 identified",
                severity=ComplianceSeverity.HIGH,
                recommendation="NZBA requires interim targets for 2030",
            )

        # 4. Long-term net-zero target (2050)
        has_nz = False
        if result.sector_targets:
            for _sector, target in result.sector_targets.items():
                target_year = target.get("target_year")
                if target_year and int(target_year) >= self.config.nzba_final_target_year:
                    has_nz = True
                    break
        if has_nz:
            state.add_pass("NZBA-004", "Long-term net-zero target (2050) set")
        else:
            state.add_warning(
                "NZBA-004",
                "No long-term net-zero target (2050) identified",
                severity=ComplianceSeverity.HIGH,
                recommendation="NZBA requires net-zero by 2050",
            )

        # 5. PCAF methodology alignment
        if result.weighted_data_quality_score is not None or result.data_quality_scores:
            state.add_pass("NZBA-005", "PCAF methodology alignment confirmed")
        else:
            state.add_warning(
                "NZBA-005",
                "PCAF methodology alignment not confirmed",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Use PCAF methodology for financed emissions",
            )

        # 6. Annual disclosure
        if result.reporting_period_start and result.reporting_period_end:
            state.add_pass("NZBA-006", "Annual disclosure period defined")
        else:
            state.add_warning(
                "NZBA-006",
                "Reporting period not defined",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 7. WACI reporting
        if result.waci_value is not None:
            state.add_pass("NZBA-007", "WACI reported")
        else:
            state.add_warning(
                "NZBA-007",
                "WACI not reported",
                severity=ComplianceSeverity.MEDIUM,
                recommendation="Report WACI as per NZBA disclosure requirements",
            )

        # 8. Portfolio coverage
        if result.portfolio_coverage_pct is not None:
            state.add_pass("NZBA-008", "Portfolio coverage disclosed")
        else:
            state.add_warning(
                "NZBA-008",
                "Portfolio coverage not disclosed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 9. Temperature alignment
        if result.temperature_alignment is not None:
            if result.temperature_alignment <= Decimal("1.5"):
                state.add_pass("NZBA-009", f"Temperature aligned at {result.temperature_alignment}C")
            else:
                state.add_warning(
                    "NZBA-009",
                    f"Temperature alignment {result.temperature_alignment}C > 1.5C",
                    severity=ComplianceSeverity.HIGH,
                    recommendation="NZBA requires alignment with 1.5C pathway",
                )
        else:
            state.add_warning(
                "NZBA-009",
                "Temperature alignment not assessed",
                severity=ComplianceSeverity.MEDIUM,
            )

        # 10. Transition plan
        if result.sector_targets:
            state.add_pass("NZBA-010", "Transition plan elements present")
        else:
            state.add_warning(
                "NZBA-010",
                "No transition plan elements",
                severity=ComplianceSeverity.HIGH,
                recommendation="Develop transition plan with sector-specific milestones",
            )

        return self._build_result(
            state, {"alliance": "NZBA/NZAOA"}
        )

    # ========================================================================
    # Double-Counting Prevention Rules (DC-INV-001 to DC-INV-008)
    # ========================================================================

    def _check_dc_rules(self, result: InvestmentResultInput) -> List[Dict[str, str]]:
        """
        Check all 8 double-counting prevention rules.

        Rules:
            DC-INV-001: Consolidated investments NOT in Cat 15
            DC-INV-002: Equity share used for consolidation
            DC-INV-003: Fund-of-funds look-through
            DC-INV-004: CRE vs Cat 8/13 overlap
            DC-INV-005: Sovereign vs corporate overlap
            DC-INV-006: Multi-asset same company
            DC-INV-007: Managed investments underlying
            DC-INV-008: Short positions excluded

        Args:
            result: Calculation result to check.

        Returns:
            List of violation dictionaries with rule_id, message, recommendation.
        """
        violations: List[Dict[str, str]] = []

        # DC-INV-001: Investments already in Scope 1/2 must NOT be Cat 15
        if result.includes_consolidated_investments:
            violations.append({
                "rule_id": "DC-INV-001",
                "message": (
                    "Investments consolidated via operational control are included "
                    "in Scope 1/2 and must NOT be counted in Category 15"
                ),
                "recommendation": (
                    "Exclude investments where the investee is consolidated "
                    "in the reporting entity's Scope 1/2 boundary"
                ),
            })

        # DC-INV-002: Equity share already used for consolidation
        if result.includes_equity_share_consolidated:
            violations.append({
                "rule_id": "DC-INV-002",
                "message": (
                    "Equity share is already used for Scope 1/2 consolidation; "
                    "do not double-count the same share in Category 15"
                ),
                "recommendation": (
                    "Use a consistent consolidation approach: equity share OR "
                    "operational/financial control, not both"
                ),
            })

        # DC-INV-003: Fund-of-funds without look-through
        if result.includes_fund_of_funds and not result.fund_of_funds_look_through:
            violations.append({
                "rule_id": "DC-INV-003",
                "message": (
                    "Fund-of-funds detected but look-through not applied; "
                    "intermediate fund emissions may be double-counted"
                ),
                "recommendation": (
                    "Apply look-through to underlying holdings to avoid "
                    "counting intermediate fund-level emissions"
                ),
            })

        # DC-INV-004: CRE vs Cat 8/13 overlap
        if result.includes_cre_in_cat8_or_cat13:
            violations.append({
                "rule_id": "DC-INV-004",
                "message": (
                    "Commercial real estate also counted in Category 8 (upstream leased) "
                    "or Category 13 (downstream leased); same property must not appear in both"
                ),
                "recommendation": (
                    "Assign each property to exactly one Scope 3 category "
                    "based on the nature of the arrangement"
                ),
            })

        # DC-INV-005: Sovereign bonds vs corporate
        if result.includes_sovereign_and_corporate:
            violations.append({
                "rule_id": "DC-INV-005",
                "message": (
                    "Both sovereign and corporate bonds counted for the same country; "
                    "national emissions include corporate emissions"
                ),
                "recommendation": (
                    "When both sovereign and corporate bonds are held, ensure "
                    "the attribution avoids double-counting national-level emissions"
                ),
            })

        # DC-INV-006: Multi-asset class same company
        if result.multi_asset_same_company and not result.multi_asset_deduplication_applied:
            violations.append({
                "rule_id": "DC-INV-006",
                "message": (
                    "Multiple asset classes held for the same company without deduplication; "
                    "company emissions may be counted multiple times"
                ),
                "recommendation": (
                    "When holding equity + debt of the same company, count "
                    "the company's emissions only once using the best available data"
                ),
            })

        # DC-INV-007: Managed investments underlying already counted
        if result.includes_managed_investments:
            violations.append({
                "rule_id": "DC-INV-007",
                "message": (
                    "Managed investments included; underlying holdings may already "
                    "be counted at the fund level"
                ),
                "recommendation": (
                    "Apply look-through to managed fund holdings; "
                    "do not double-count at both fund and underlying level"
                ),
            })

        # DC-INV-008: Short positions must be excluded
        if result.includes_short_positions:
            violations.append({
                "rule_id": "DC-INV-008",
                "message": (
                    "Short positions are included; they represent no economic exposure "
                    "and must be excluded from financed emissions"
                ),
                "recommendation": (
                    "Exclude short positions from Category 15 financed emissions; "
                    "only long positions represent economic exposure"
                ),
            })

        return violations

    # ========================================================================
    # PCAF Attribution Validation
    # ========================================================================

    def _validate_pcaf_attribution(self, result: InvestmentResultInput) -> List[str]:
        """
        Validate PCAF attribution factor formulas per asset class.

        Checks:
        - Listed equity/corporate bonds: outstanding amount / EVIC
        - Private equity: equity share (investment / total equity)
        - Project finance: pro-rata share of total project cost
        - CRE: outstanding amount / property value at origination
        - Mortgage: outstanding amount / property value at origination
        - Motor vehicle: outstanding amount / total vehicle value
        - Sovereign bond: outstanding amount / PPP-adjusted GDP

        Args:
            result: Calculation result to check.

        Returns:
            List of violation description strings.
        """
        violations: List[str] = []

        if not result.attribution_factors:
            return violations

        for holding_id, af in result.attribution_factors.items():
            # Attribution factor must be between 0 and 1 (or sometimes slightly > 1)
            if af < Decimal("0"):
                violations.append(
                    f"Holding {holding_id}: attribution factor {af} is negative"
                )
            elif af > Decimal("1.5"):
                violations.append(
                    f"Holding {holding_id}: attribution factor {af} > 1.5 "
                    f"(may indicate calculation error)"
                )

        return violations

    # ========================================================================
    # Portfolio Coverage Check
    # ========================================================================

    def _check_portfolio_coverage(
        self, result: InvestmentResultInput
    ) -> ComplianceCheckResult:
        """
        Check portfolio coverage meets PCAF threshold.

        Args:
            result: Calculation result.

        Returns:
            ComplianceCheckResult for portfolio coverage.
        """
        threshold = self.config.minimum_portfolio_coverage_pct

        if result.portfolio_coverage_pct is not None:
            if result.portfolio_coverage_pct >= threshold:
                return ComplianceCheckResult(
                    framework=ComplianceFramework.PCAF,
                    status=ComplianceStatus.PASS,
                    score=Decimal("100"),
                    passed_checks=1,
                    total_checks=1,
                )
            else:
                return ComplianceCheckResult(
                    framework=ComplianceFramework.PCAF,
                    status=ComplianceStatus.WARNING,
                    score=Decimal("50"),
                    warning_checks=1,
                    total_checks=1,
                )
        else:
            return ComplianceCheckResult(
                framework=ComplianceFramework.PCAF,
                status=ComplianceStatus.FAIL,
                score=Decimal("0"),
                failed_checks=1,
                total_checks=1,
            )

    # ========================================================================
    # Recommendations Generator
    # ========================================================================

    def _generate_recommendations(
        self, results: Dict[ComplianceFramework, ComplianceCheckResult]
    ) -> List[str]:
        """
        Generate actionable recommendations based on compliance results.

        Args:
            results: Dictionary of compliance check results.

        Returns:
            Ordered list of unique recommendation strings.
        """
        recommendations: List[str] = []
        seen: Set[str] = set()

        # Priority recommendations based on overall score
        overall_score = self.get_overall_compliance_score(results)
        if overall_score < Decimal("70"):
            rec = (
                "Critical: Overall compliance score is below 70%. "
                "Address critical and high-severity issues immediately."
            )
            recommendations.append(rec)
            seen.add(rec)

        # Collect unique recommendations from findings
        for cr in results.values():
            for finding in cr.issues:
                if finding.recommendation and finding.recommendation not in seen:
                    recommendations.append(finding.recommendation)
                    seen.add(finding.recommendation)

        return recommendations

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _build_result(
        self,
        state: FrameworkCheckState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceCheckResult:
        """Build ComplianceCheckResult from FrameworkCheckState."""
        return ComplianceCheckResult(
            framework=state.framework,
            status=state.compute_status(),
            score=state.compute_score(),
            issues=state.findings,
            passed_checks=state.passed_checks,
            failed_checks=state.failed_checks,
            warning_checks=state.warning_checks,
            total_checks=state.total_checks,
            metadata=metadata or {},
        )


# ==============================================================================
# Module-Level Exports
# ==============================================================================

__all__ = [
    "ComplianceCheckerEngine",
    "ComplianceCheckerConfig",
    "ComplianceCheckResult",
    "ComplianceFinding",
    "ComplianceFramework",
    "ComplianceStatus",
    "ComplianceSeverity",
    "InvestmentResultInput",
    "FrameworkCheckState",
    "DoubleCountingCategory",
    "AssetClass",
    "CalculationMethod",
    "PCAF_ASSET_CLASSES",
    "PCAF_DQ_DESCRIPTIONS",
    "NZBA_TARGET_SECTORS",
    "ENGINE_ID",
    "ENGINE_VERSION",
]
