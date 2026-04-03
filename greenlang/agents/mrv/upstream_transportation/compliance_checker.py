"""
ComplianceCheckerEngine - AGENT-MRV-017 Engine 6

This module implements the ComplianceCheckerEngine for Upstream Transportation (Category 4).
It validates calculations against 7 regulatory frameworks with Category 4-specific compliance rules.

Regulatory Frameworks:
1. GHG Protocol Scope 3 (Category 4 specific)
2. ISO 14083:2023 (transport-specific, WTW mandatory)
3. GLEC Framework v3.2
4. CSRD/ESRS E1
5. CDP Climate Change
6. SBTi
7. GRI 305

Category 4-Specific Compliance Rules:
- Payment boundary enforcement (Incoterms)
- Well-to-Wheel vs Tank-to-Wheel scope
- Double-counting prevention (Cat 1, 3, 9)
- Transport chain completeness
- Data quality minimum thresholds

Example:
    >>> engine = ComplianceCheckerEngine(config)
    >>> result = await engine.check_all_frameworks(calculation_result)
    >>> overall_score = engine.get_overall_compliance_score(result)
    >>> print(f"Compliance: {overall_score}%")
"""

from typing import Dict, List, Optional, Any, Set
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class ComplianceFramework(str, Enum):
    """Supported regulatory frameworks."""

    GHG_PROTOCOL = "GHG_PROTOCOL"
    ISO_14083 = "ISO_14083"
    GLEC_FRAMEWORK = "GLEC_FRAMEWORK"
    CSRD_ESRS_E1 = "CSRD_ESRS_E1"
    CDP = "CDP"
    SBTI = "SBTI"
    GRI_305 = "GRI_305"


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComplianceSeverity(str, Enum):
    """Severity level for compliance issues."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Incoterm(str, Enum):
    """Incoterms 2020 classification."""

    # Category 4 (Company pays transport)
    DDP = "DDP"  # Delivered Duty Paid
    CIF = "CIF"  # Cost, Insurance, and Freight
    CIP = "CIP"  # Carriage and Insurance Paid To
    CPT = "CPT"  # Carriage Paid To
    DAP = "DAP"  # Delivered at Place
    DPU = "DPU"  # Delivered at Place Unloaded

    # Category 9 (Supplier pays transport)
    EXW = "EXW"  # Ex Works
    FCA = "FCA"  # Free Carrier
    FAS = "FAS"  # Free Alongside Ship
    FOB = "FOB"  # Free on Board
    CFR = "CFR"  # Cost and Freight


class EmissionScope(str, Enum):
    """Emission scope for transport."""

    TTW = "TTW"  # Tank-to-Wheel
    WTW = "WTW"  # Well-to-Wheel
    WTT = "WTT"  # Well-to-Tank


class TransportMode(str, Enum):
    """Transport mode classification."""

    ROAD = "ROAD"
    RAIL = "RAIL"
    SEA = "SEA"
    AIR = "AIR"
    INLAND_WATERWAY = "INLAND_WATERWAY"
    PIPELINE = "PIPELINE"
    MULTIMODAL = "MULTIMODAL"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ComplianceIssue:
    """Single compliance issue."""

    rule_id: str
    severity: ComplianceSeverity
    status: ComplianceStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    regulation_reference: Optional[str] = None


@dataclass
class ComplianceCheckResult:
    """Result of compliance check for one framework."""

    framework: ComplianceFramework
    status: ComplianceStatus
    score: Decimal  # 0-100
    issues: List[ComplianceIssue] = field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_checks: int = 0
    checked_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CalculationResultInput(BaseModel):
    """Input calculation result for compliance checking."""

    total_emissions_kg_co2e: Decimal = Field(..., description="Total emissions")
    co2_kg: Decimal = Field(default=Decimal("0"), description="CO2 emissions")
    ch4_kg: Decimal = Field(default=Decimal("0"), description="CH4 emissions")
    n2o_kg: Decimal = Field(default=Decimal("0"), description="N2O emissions")

    emission_scope: EmissionScope = Field(..., description="WTW or TTW")
    calculation_method: str = Field(..., description="Method used")

    # Shipment details
    shipments: List[Dict[str, Any]] = Field(default_factory=list)
    transport_modes: List[TransportMode] = Field(default_factory=list)
    incoterms: List[Incoterm] = Field(default_factory=list)

    # Data quality
    data_quality_score: Optional[Decimal] = Field(None, ge=0, le=5)
    spend_based_percentage: Optional[Decimal] = Field(None, ge=0, le=100)

    # Allocation
    allocation_method: Optional[str] = None
    allocation_methods_used: List[str] = Field(default_factory=list)

    # Boundaries
    includes_reefer: bool = Field(default=False)
    includes_warehousing: bool = Field(default=False)
    includes_multi_leg: bool = Field(default=False)

    # Reporting
    reporting_period_start: Optional[datetime] = None
    reporting_period_end: Optional[datetime] = None
    base_year: Optional[int] = None

    # Uncertainty
    uncertainty_percentage: Optional[Decimal] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceCheckerConfig(BaseModel):
    """Configuration for ComplianceCheckerEngine."""

    # Framework enablement
    enabled_frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: list(ComplianceFramework)
    )

    # Thresholds
    minimum_data_quality_score: Decimal = Field(
        default=Decimal("2.0"),
        ge=0,
        le=5,
        description="Minimum DQI score (0-5 scale)"
    )
    maximum_spend_based_percentage: Decimal = Field(
        default=Decimal("80.0"),
        ge=0,
        le=100,
        description="Max % of emissions from spend-based method"
    )
    minimum_mode_coverage_percentage: Decimal = Field(
        default=Decimal("95.0"),
        ge=0,
        le=100,
        description="Min % of spend/volume covered by modes"
    )

    # SBTi-specific
    sbti_scope3_coverage_threshold: Decimal = Field(
        default=Decimal("67.0"),
        ge=0,
        le=100,
        description="SBTi requires 67% Scope 3 coverage"
    )

    # Tolerances
    allow_ttw_for_non_iso: bool = Field(
        default=True,
        description="Allow TTW for frameworks other than ISO 14083"
    )
    require_multi_leg_for_intercontinental: bool = Field(
        default=True,
        description="Require multi-leg for intercontinental routes"
    )

    # Scoring weights
    critical_weight: Decimal = Field(default=Decimal("1.0"))
    high_weight: Decimal = Field(default=Decimal("0.8"))
    medium_weight: Decimal = Field(default=Decimal("0.5"))
    low_weight: Decimal = Field(default=Decimal("0.2"))
    info_weight: Decimal = Field(default=Decimal("0.0"))


# ============================================================================
# ComplianceCheckerEngine
# ============================================================================


class ComplianceCheckerEngine:
    """
    ComplianceCheckerEngine - validates calculations against 7 regulatory frameworks.

    This engine implements comprehensive compliance checking for Upstream Transportation
    (Category 4) against major regulatory frameworks including GHG Protocol, ISO 14083,
    GLEC Framework, CSRD/ESRS E1, CDP, SBTi, and GRI 305.

    Category 4-Specific Compliance Rules:
    1. Payment boundary enforcement (Incoterms)
    2. Well-to-Wheel vs Tank-to-Wheel scope
    3. Double-counting prevention (Cat 1, 3, 9)
    4. Transport chain completeness
    5. Data quality minimum thresholds
    6. Mode coverage requirements
    7. Reefer and warehousing inclusion
    8. Allocation method consistency
    9. Boundary documentation
    10. Uncertainty quantification

    Attributes:
        config: Engine configuration
        category_4_incoterms: Set of Incoterms that belong to Category 4
        category_9_incoterms: Set of Incoterms that belong to Category 9

    Example:
        >>> config = ComplianceCheckerConfig()
        >>> engine = ComplianceCheckerEngine(config)
        >>> result_input = CalculationResultInput(...)
        >>> all_results = engine.check_all_frameworks(result_input)
        >>> overall_score = engine.get_overall_compliance_score(all_results)
    """

    def __init__(self, config: ComplianceCheckerConfig):
        """Initialize ComplianceCheckerEngine."""
        self.config = config

        # Incoterm classification
        self.category_4_incoterms = {
            Incoterm.DDP, Incoterm.CIF, Incoterm.CIP,
            Incoterm.CPT, Incoterm.DAP, Incoterm.DPU
        }
        self.category_9_incoterms = {
            Incoterm.EXW, Incoterm.FCA, Incoterm.FAS,
            Incoterm.FOB, Incoterm.CFR
        }

        logger.info("ComplianceCheckerEngine initialized with %s frameworks", len(config.enabled_frameworks))

    # ========================================================================
    # Main Entry Points
    # ========================================================================

    def check_all_frameworks(
        self,
        result: CalculationResultInput
    ) -> Dict[ComplianceFramework, ComplianceCheckResult]:
        """
        Check compliance against all enabled frameworks.

        Args:
            result: Calculation result to validate

        Returns:
            Dictionary mapping framework to compliance check result

        Example:
            >>> results = engine.check_all_frameworks(calc_result)
            >>> for framework, check_result in results.items():
            ...     print(f"{framework}: {check_result.status}")
        """
        logger.info("Running compliance checks for all frameworks")

        all_results = {}

        for framework in self.config.enabled_frameworks:
            try:
                if framework == ComplianceFramework.GHG_PROTOCOL:
                    check_result = self.check_ghg_protocol(result)
                elif framework == ComplianceFramework.ISO_14083:
                    check_result = self.check_iso_14083(result)
                elif framework == ComplianceFramework.GLEC_FRAMEWORK:
                    check_result = self.check_glec_framework(result)
                elif framework == ComplianceFramework.CSRD_ESRS_E1:
                    check_result = self.check_csrd_esrs_e1(result)
                elif framework == ComplianceFramework.CDP:
                    check_result = self.check_cdp(result)
                elif framework == ComplianceFramework.SBTI:
                    check_result = self.check_sbti(result)
                elif framework == ComplianceFramework.GRI_305:
                    check_result = self.check_gri_305(result)
                else:
                    logger.warning("Unknown framework: %s", framework)
                    continue

                all_results[framework] = check_result
                logger.info(
                    f"{framework} compliance: {check_result.status} "
                    f"(score: {check_result.score})"
                )

            except Exception as e:
                logger.error(
                    f"Error checking {framework} compliance: {str(e)}",
                    exc_info=True
                )
                # Create failed result
                all_results[framework] = ComplianceCheckResult(
                    framework=framework,
                    status=ComplianceStatus.FAIL,
                    score=Decimal("0"),
                    issues=[
                        ComplianceIssue(
                            rule_id="CHECK_ERROR",
                            severity=ComplianceSeverity.CRITICAL,
                            status=ComplianceStatus.FAIL,
                            message=f"Compliance check failed: {str(e)}"
                        )
                    ]
                )

        return all_results

    def get_overall_compliance_score(
        self,
        results: Dict[ComplianceFramework, ComplianceCheckResult]
    ) -> Decimal:
        """
        Calculate overall compliance score across all frameworks.

        Args:
            results: Dictionary of compliance check results

        Returns:
            Overall compliance score (0-100)

        Example:
            >>> score = engine.get_overall_compliance_score(results)
            >>> print(f"Overall compliance: {score}%")
        """
        if not results:
            return Decimal("0")

        total_score = sum(r.score for r in results.values())
        avg_score = total_score / len(results)

        return avg_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def get_compliance_summary(
        self,
        results: Dict[ComplianceFramework, ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Generate compliance summary across all frameworks.

        Args:
            results: Dictionary of compliance check results

        Returns:
            Summary dictionary with scores, issues, recommendations

        Example:
            >>> summary = engine.get_compliance_summary(results)
            >>> print(summary["overall_status"])
        """
        overall_score = self.get_overall_compliance_score(results)

        # Determine overall status
        if overall_score >= 95:
            overall_status = ComplianceStatus.PASS
        elif overall_score >= 70:
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.FAIL

        # Aggregate issues by severity
        critical_issues = []
        high_issues = []
        medium_issues = []
        low_issues = []

        for framework, result in results.items():
            for issue in result.issues:
                issue_with_framework = {
                    "framework": framework.value,
                    "rule_id": issue.rule_id,
                    "message": issue.message,
                    "recommendation": issue.recommendation
                }

                if issue.severity == ComplianceSeverity.CRITICAL:
                    critical_issues.append(issue_with_framework)
                elif issue.severity == ComplianceSeverity.HIGH:
                    high_issues.append(issue_with_framework)
                elif issue.severity == ComplianceSeverity.MEDIUM:
                    medium_issues.append(issue_with_framework)
                elif issue.severity == ComplianceSeverity.LOW:
                    low_issues.append(issue_with_framework)

        # Framework-specific scores
        framework_scores = {
            framework.value: {
                "score": float(result.score),
                "status": result.status.value,
                "passed": result.passed_checks,
                "failed": result.failed_checks,
                "warnings": result.warning_checks
            }
            for framework, result in results.items()
        }

        return {
            "overall_score": float(overall_score),
            "overall_status": overall_status.value,
            "frameworks_checked": len(results),
            "framework_scores": framework_scores,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "low_issues": low_issues,
            "total_issues": len(critical_issues) + len(high_issues) + len(medium_issues) + len(low_issues),
            "recommendations": self.get_recommendations(results)
        }

    def get_recommendations(
        self,
        results: Dict[ComplianceFramework, ComplianceCheckResult]
    ) -> List[str]:
        """
        Generate actionable recommendations based on compliance results.

        Args:
            results: Dictionary of compliance check results

        Returns:
            List of recommendation strings

        Example:
            >>> recommendations = engine.get_recommendations(results)
            >>> for rec in recommendations:
            ...     print(f"- {rec}")
        """
        recommendations = []
        seen_recommendations = set()

        # Collect unique recommendations from all issues
        for result in results.values():
            for issue in result.issues:
                if issue.recommendation and issue.recommendation not in seen_recommendations:
                    recommendations.append(issue.recommendation)
                    seen_recommendations.add(issue.recommendation)

        # Add general recommendations based on overall compliance
        overall_score = self.get_overall_compliance_score(results)

        if overall_score < 70:
            recommendations.insert(
                0,
                "Critical: Overall compliance score is below 70%. "
                "Address critical and high-severity issues immediately."
            )

        return recommendations

    # ========================================================================
    # GHG Protocol Scope 3 Compliance
    # ========================================================================

    def check_ghg_protocol(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 (Category 4 specific).

        GHG Protocol Requirements:
        1. Payment boundary (Incoterms-based category assignment)
        2. Double-counting prevention (Cat 1, 3, 9)
        3. Data quality assessment
        4. Uncertainty quantification
        5. Methodology documentation
        6. Transport chain completeness
        7. Allocation method disclosure
        8. Consolidation approach
        9. Base year consistency
        10. Reporting period definition
        11. Materiality assessment
        12. Exclusion justification
        13. Emission factor sources
        14. Multi-leg journey tracking
        15. Third-party verification readiness

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for GHG Protocol

        Example:
            >>> ghg_result = engine.check_ghg_protocol(calc_result)
            >>> print(ghg_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Payment boundary check
        issue = self._check_payment_boundary(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            elif issue.status == ComplianceStatus.WARNING:
                warnings += 1
        else:
            passed += 1

        # 2. Incoterms classification
        issue = self._check_incoterms_classification(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 3. Double-counting prevention (Cat 1)
        issue = self._check_double_counting_cat1(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Double-counting prevention (Cat 3)
        issue = self._check_double_counting_cat3(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Double-counting prevention (Cat 9)
        issue = self._check_double_counting_cat9(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 6. Data quality minimum
        issue = self._check_data_quality_minimum(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Transport chain completeness
        issue = self._check_transport_chain_completeness(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Allocation method consistency
        issue = self._check_allocation_method_consistency(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 9. Uncertainty quantification
        issue = self._check_uncertainty_quantification(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 11. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 12. Mode coverage
        issue = self._check_mode_coverage(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 13. Reefer inclusion
        issue = self._check_reefer_inclusion(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 14. Warehousing inclusion
        issue = self._check_warehousing_inclusion(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 15. Emission breakdown
        issue = self._check_emission_breakdown(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)

        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.GHG_PROTOCOL,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # ISO 14083:2023 Compliance
    # ========================================================================

    def check_iso_14083(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14083:2023 (transport-specific standard).

        ISO 14083 Requirements:
        1. Well-to-Wheel (WTW) scope MANDATORY
        2. Transport chain concept (all legs and hubs)
        3. Transport Chain Element (TCE) identification
        4. Default values usage documentation
        5. Allocation per ISO 14083 Annex B
        6. Data quality requirements
        7. Uncertainty assessment
        8. Multi-modal transport handling
        9. Empty running consideration
        10. Load factor documentation
        11. Energy carrier specification
        12. Emission factor sources (ISO-compliant)

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for ISO 14083

        Example:
            >>> iso_result = engine.check_iso_14083(calc_result)
            >>> print(iso_result.status)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. WTW mandatory for ISO 14083
        issue = self._check_wtw_mandatory(result)
        if issue:
            issues.append(issue)
            failed += 1
        else:
            passed += 1

        # 2. Transport chain completeness
        issue = self._check_transport_chain_completeness(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 3. TCE identification
        issue = self._check_tce_identification(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Default values documentation
        issue = self._check_default_values_documentation(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Allocation method
        issue = self._check_allocation_method_consistency(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Data quality
        issue = self._check_data_quality_minimum(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Uncertainty
        issue = self._check_uncertainty_quantification(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Multi-modal handling
        issue = self._check_multimodal_handling(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 9. Empty running
        issue = self._check_empty_running_consideration(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Load factor
        issue = self._check_load_factor_documentation(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 11. Energy carrier
        issue = self._check_energy_carrier_specification(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 12. Emission factor sources
        issue = self._check_emission_factor_sources(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)
        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.ISO_14083,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # GLEC Framework Compliance
    # ========================================================================

    def check_glec_framework(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with GLEC Framework v3.2.

        GLEC Framework Requirements:
        1. GLEC-accredited emission factors
        2. Default values from GLEC database
        3. Allocation methodology per GLEC
        4. Mode coverage requirements
        5. Data quality classification
        6. Hub-to-hub methodology
        7. Shipment-level tracking
        8. TCE aggregation
        9. Verification readiness
        10. Reporting format compliance

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for GLEC Framework

        Example:
            >>> glec_result = engine.check_glec_framework(calc_result)
            >>> print(glec_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. GLEC emission factors
        issue = self._check_glec_emission_factors(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Default values
        issue = self._check_default_values_documentation(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Allocation methodology
        issue = self._check_allocation_method_consistency(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Mode coverage
        issue = self._check_mode_coverage(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Data quality
        issue = self._check_data_quality_minimum(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Hub-to-hub methodology
        issue = self._check_hub_to_hub_methodology(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Shipment-level tracking
        issue = self._check_shipment_level_tracking(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. TCE aggregation
        issue = self._check_tce_aggregation(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 9. Transport chain completeness
        issue = self._check_transport_chain_completeness(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Emission breakdown
        issue = self._check_emission_breakdown(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)
        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.GLEC_FRAMEWORK,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # CSRD/ESRS E1 Compliance
    # ========================================================================

    def check_csrd_esrs_e1(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with CSRD/ESRS E1 (EU Corporate Sustainability Reporting Directive).

        CSRD/ESRS E1 Requirements:
        1. Scope 3 Category 4 disclosure
        2. Double materiality assessment
        3. Value chain mapping
        4. Data quality disclosure
        5. Estimation methodology disclosure
        6. Uncertainty disclosure
        7. Target alignment (if applicable)
        8. Transition plan integration
        9. ESRS E1-6 (Scope 3 emissions)
        10. GHG inventory completeness
        11. Baseline year consistency
        12. Reporting boundary definition

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for CSRD/ESRS E1

        Example:
            >>> csrd_result = engine.check_csrd_esrs_e1(calc_result)
            >>> print(csrd_result.status)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Scope 3 disclosure
        issue = self._check_scope3_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Double materiality
        issue = self._check_double_materiality(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Value chain mapping
        issue = self._check_value_chain_mapping(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Data quality disclosure
        issue = self._check_data_quality_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Methodology disclosure
        issue = self._check_methodology_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Uncertainty disclosure
        issue = self._check_uncertainty_quantification(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Target alignment
        issue = self._check_target_alignment(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 9. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Boundary definition
        issue = self._check_boundary_definition(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 11. Emission breakdown
        issue = self._check_emission_breakdown(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 12. Payment boundary
        issue = self._check_payment_boundary(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)
        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.CSRD_ESRS_E1,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # CDP Climate Change Compliance
    # ========================================================================

    def check_cdp(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change questionnaire.

        CDP Requirements:
        1. Scope 3 Category 4 reporting
        2. Methodology disclosure (C6.5)
        3. Data quality assessment
        4. Emission factor sources
        5. Percentage of emissions calculated vs estimated
        6. Verification status
        7. Exclusions justification
        8. Base year emissions
        9. Target coverage
        10. Supplier engagement (if applicable)

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for CDP

        Example:
            >>> cdp_result = engine.check_cdp(calc_result)
            >>> print(cdp_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Scope 3 Category 4 reporting
        issue = self._check_scope3_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Methodology disclosure
        issue = self._check_methodology_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Data quality
        issue = self._check_data_quality_minimum(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Emission factor sources
        issue = self._check_emission_factor_sources(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Calculation vs estimation percentage
        issue = self._check_calculation_vs_estimation(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Emission breakdown
        issue = self._check_emission_breakdown(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 9. Mode coverage
        issue = self._check_mode_coverage(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Payment boundary
        issue = self._check_payment_boundary(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)
        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.CDP,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # SBTi Compliance
    # ========================================================================

    def check_sbti(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative (SBTi).

        SBTi Requirements:
        1. 67% Scope 3 coverage (if Scope 3 > 40% of total)
        2. Category 4 inclusion in target boundary
        3. Base year definition
        4. Target ambition level
        5. Data quality requirements
        6. Recalculation policy
        7. GHG Protocol compliance
        8. Verification readiness

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for SBTi

        Example:
            >>> sbti_result = engine.check_sbti(calc_result)
            >>> print(sbti_result.status)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Scope 3 coverage (67% threshold)
        issue = self._check_scope3_coverage(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Category 4 in target boundary
        issue = self._check_category4_in_target(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Data quality
        issue = self._check_data_quality_minimum(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. GHG Protocol compliance (payment boundary)
        issue = self._check_payment_boundary(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Methodology disclosure
        issue = self._check_methodology_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Emission breakdown
        issue = self._check_emission_breakdown(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)
        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.SBTI,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # GRI 305 Compliance
    # ========================================================================

    def check_gri_305(
        self,
        result: CalculationResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305: Emissions.

        GRI 305 Requirements:
        1. GRI 305-3 (Other indirect GHG emissions - Scope 3)
        2. Gases included (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
        3. Emission sources (upstream transportation)
        4. Consolidation approach
        5. Methodologies and assumptions
        6. Emission factors and GWP rates
        7. Base year emissions
        8. Contextual information

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for GRI 305

        Example:
            >>> gri_result = engine.check_gri_305(calc_result)
            >>> print(gri_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. GRI 305-3 disclosure
        issue = self._check_gri_305_3_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Gases included
        issue = self._check_emission_breakdown(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Emission sources
        issue = self._check_emission_sources(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Consolidation approach
        issue = self._check_consolidation_approach(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Methodologies and assumptions
        issue = self._check_methodology_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Emission factors
        issue = self._check_emission_factor_sources(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        total = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings, issues)
        status = self._determine_status(score, failed)

        return ComplianceCheckResult(
            framework=ComplianceFramework.GRI_305,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total
        )

    # ========================================================================
    # Category 4-Specific Compliance Rules
    # ========================================================================

    def _check_payment_boundary(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """
        PAYMENT_BOUNDARY: Only company-paid transport belongs in Category 4.

        Checks that Incoterms are correctly assigned to Category 4 vs Category 9.
        """
        if not result.incoterms:
            return ComplianceIssue(
                rule_id="PAYMENT_BOUNDARY",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message="No Incoterms specified for shipments",
                recommendation="Assign Incoterms to all shipments to ensure correct category assignment (Cat 4 vs Cat 9)",
                regulation_reference="GHG Protocol Scope 3, Category 4 & 9 guidance"
            )

        # Check for Category 9 Incoterms in Category 4 calculation
        cat9_incoterms_found = [
            incoterm for incoterm in result.incoterms
            if incoterm in self.category_9_incoterms
        ]

        if cat9_incoterms_found:
            return ComplianceIssue(
                rule_id="PAYMENT_BOUNDARY",
                severity=ComplianceSeverity.CRITICAL,
                status=ComplianceStatus.FAIL,
                message=f"Category 9 Incoterms found in Category 4 calculation: {cat9_incoterms_found}",
                details={"invalid_incoterms": [i.value for i in cat9_incoterms_found]},
                recommendation="Remove supplier-paid transport (EXW, FCA, FAS, FOB, CFR) from Category 4; report in Category 9",
                regulation_reference="GHG Protocol Scope 3, Category 4 & 9 payment boundary"
            )

        return None

    def _check_incoterms_classification(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """INCOTERMS_CLASSIFICATION: Verify Incoterm-to-category mapping."""
        if not result.shipments:
            return ComplianceIssue(
                rule_id="INCOTERMS_CLASSIFICATION",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="No shipment-level data available to verify Incoterm classification",
                recommendation="Track Incoterms at shipment level for accurate category assignment"
            )

        shipments_without_incoterm = [
            s for s in result.shipments
            if not s.get("incoterm")
        ]

        if shipments_without_incoterm:
            return ComplianceIssue(
                rule_id="INCOTERMS_CLASSIFICATION",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message=f"{len(shipments_without_incoterm)} shipments missing Incoterm assignment",
                details={"count": len(shipments_without_incoterm)},
                recommendation="Assign Incoterms to all shipments; document default assumption if not available"
            )

        return None

    def _check_mode_coverage(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """MODE_COVERAGE: All material transport modes included."""
        if not result.transport_modes:
            return ComplianceIssue(
                rule_id="MODE_COVERAGE",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message="No transport modes specified",
                recommendation="Include all material transport modes (road, rail, sea, air, etc.)",
                regulation_reference="GHG Protocol Scope 3, Category 4 - completeness"
            )

        # Check for single-mode reporting (often incomplete)
        if len(result.transport_modes) == 1:
            return ComplianceIssue(
                rule_id="MODE_COVERAGE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message=f"Only one transport mode reported: {result.transport_modes[0].value}",
                recommendation="Verify that all material transport modes are included (>95% of spend/volume)",
                regulation_reference="GHG Protocol Scope 3 - materiality threshold"
            )

        # Check mode coverage percentage if available
        if result.metadata.get("mode_coverage_percentage"):
            coverage = Decimal(str(result.metadata["mode_coverage_percentage"]))
            if coverage < self.config.minimum_mode_coverage_percentage:
                return ComplianceIssue(
                    rule_id="MODE_COVERAGE",
                    severity=ComplianceSeverity.MEDIUM,
                    status=ComplianceStatus.WARNING,
                    message=f"Mode coverage is {coverage}%, below minimum {self.config.minimum_mode_coverage_percentage}%",
                    details={"coverage_percentage": float(coverage)},
                    recommendation="Expand mode coverage to include >95% of transport spend/volume"
                )

        return None

    def _check_wtw_mandatory(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """WTW_MANDATORY: ISO 14083 requires Well-to-Wheel scope."""
        if result.emission_scope != EmissionScope.WTW:
            return ComplianceIssue(
                rule_id="WTW_MANDATORY",
                severity=ComplianceSeverity.CRITICAL,
                status=ComplianceStatus.FAIL,
                message=f"ISO 14083 requires Well-to-Wheel (WTW) scope, but {result.emission_scope.value} was used",
                details={"current_scope": result.emission_scope.value},
                recommendation="Use Well-to-Wheel (WTW) emission factors for ISO 14083 compliance",
                regulation_reference="ISO 14083:2023, Section 6.3.2 - Scope of emissions"
            )

        return None

    def _check_transport_chain_completeness(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """TRANSPORT_CHAIN_COMPLETENESS: All legs and hubs accounted."""
        if not result.includes_multi_leg and self.config.require_multi_leg_for_intercontinental:
            # Check if any shipments are likely intercontinental
            intercontinental_likely = False
            for shipment in result.shipments:
                if shipment.get("transport_mode") == TransportMode.AIR.value:
                    intercontinental_likely = True
                    break
                if shipment.get("transport_mode") == TransportMode.SEA.value:
                    intercontinental_likely = True
                    break

            if intercontinental_likely:
                return ComplianceIssue(
                    rule_id="TRANSPORT_CHAIN_COMPLETENESS",
                    severity=ComplianceSeverity.MEDIUM,
                    status=ComplianceStatus.WARNING,
                    message="Intercontinental shipments detected but multi-leg journeys not tracked",
                    recommendation="Track all legs of intercontinental transport chains (e.g., factory → port → port → distribution center)",
                    regulation_reference="ISO 14083:2023, Transport chain concept"
                )

        return None

    def _check_allocation_method_consistency(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """ALLOCATION_METHOD_CONSISTENCY: Same allocation method within transport chain."""
        if len(result.allocation_methods_used) > 1:
            return ComplianceIssue(
                rule_id="ALLOCATION_METHOD_CONSISTENCY",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message=f"Multiple allocation methods used: {result.allocation_methods_used}",
                details={"methods": result.allocation_methods_used},
                recommendation="Use consistent allocation method (mass, volume, or revenue) across transport chain legs",
                regulation_reference="ISO 14083:2023, Annex B - Allocation"
            )

        return None

    def _check_double_counting_cat1(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """DOUBLE_COUNTING_CAT1: No overlap with Category 1 cradle-to-gate factors."""
        # Check metadata for boundary documentation
        if not result.metadata.get("cat1_boundary_documented"):
            return ComplianceIssue(
                rule_id="DOUBLE_COUNTING_CAT1",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Boundary between Category 1 and Category 4 not documented",
                recommendation="Document whether Category 1 uses cradle-to-gate or gate-to-gate emission factors to avoid double-counting transport",
                regulation_reference="GHG Protocol Scope 3, Double-counting guidance"
            )

        return None

    def _check_double_counting_cat3(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """DOUBLE_COUNTING_CAT3: WTT scope consistency with Category 3."""
        if result.emission_scope == EmissionScope.WTW:
            if not result.metadata.get("cat3_wtt_excluded"):
                return ComplianceIssue(
                    rule_id="DOUBLE_COUNTING_CAT3",
                    severity=ComplianceSeverity.MEDIUM,
                    status=ComplianceStatus.WARNING,
                    message="Using WTW for Category 4; ensure WTT of transport fuel not double-counted in Category 3",
                    recommendation="If using WTW (Well-to-Wheel) for Category 4, exclude Well-to-Tank emissions of transport fuel from Category 3 (Fuel & Energy)",
                    regulation_reference="GHG Protocol Scope 3, Category 3 & 4 boundary"
                )

        return None

    def _check_double_counting_cat9(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """DOUBLE_COUNTING_CAT9: Payment boundary enforcement."""
        # This is checked in _check_payment_boundary
        # Additional check: ensure no shipment appears in both Cat 4 and Cat 9
        if result.metadata.get("potential_cat9_overlap"):
            return ComplianceIssue(
                rule_id="DOUBLE_COUNTING_CAT9",
                severity=ComplianceSeverity.CRITICAL,
                status=ComplianceStatus.FAIL,
                message="Potential overlap detected between Category 4 and Category 9",
                recommendation="Ensure each shipment is reported in ONLY one category (Cat 4 OR Cat 9, not both)",
                regulation_reference="GHG Protocol Scope 3, Category 4 & 9 mutual exclusivity"
            )

        return None

    def _check_reefer_inclusion(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """REEFER_INCLUSION: Temperature-controlled transport emissions included."""
        if result.metadata.get("company_ships_temperature_sensitive"):
            if not result.includes_reefer:
                return ComplianceIssue(
                    rule_id="REEFER_INCLUSION",
                    severity=ComplianceSeverity.MEDIUM,
                    status=ComplianceStatus.WARNING,
                    message="Company ships temperature-sensitive goods but reefer emissions not included",
                    recommendation="Include refrigeration unit emissions for temperature-controlled transport (reefer containers, refrigerated trucks)",
                    regulation_reference="GLEC Framework v3.2 - Reefer emissions"
                )

        return None

    def _check_warehousing_inclusion(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """WAREHOUSING_INCLUSION: Third-party storage emissions included."""
        if result.metadata.get("uses_3pl_warehousing"):
            if not result.includes_warehousing:
                return ComplianceIssue(
                    rule_id="WAREHOUSING_INCLUSION",
                    severity=ComplianceSeverity.MEDIUM,
                    status=ComplianceStatus.WARNING,
                    message="Company uses 3PL warehousing but warehousing emissions not included",
                    recommendation="Include emissions from third-party warehousing and storage in Category 4",
                    regulation_reference="GHG Protocol Scope 3, Category 4 - warehousing"
                )

        return None

    def _check_data_quality_minimum(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """DATA_QUALITY_MINIMUM: Method hierarchy compliance."""
        issues = []

        # Check DQI score
        if result.data_quality_score is not None:
            if result.data_quality_score < self.config.minimum_data_quality_score:
                issues.append(
                    f"DQI score {result.data_quality_score} is below minimum {self.config.minimum_data_quality_score}"
                )

        # Check spend-based percentage
        if result.spend_based_percentage is not None:
            if result.spend_based_percentage > self.config.maximum_spend_based_percentage:
                issues.append(
                    f"Spend-based method accounts for {result.spend_based_percentage}% of emissions, "
                    f"above maximum {self.config.maximum_spend_based_percentage}%"
                )

        if issues:
            return ComplianceIssue(
                rule_id="DATA_QUALITY_MINIMUM",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Data quality below recommended thresholds",
                details={
                    "dqi_score": float(result.data_quality_score) if result.data_quality_score else None,
                    "spend_based_percentage": float(result.spend_based_percentage) if result.spend_based_percentage else None
                },
                recommendation="Upgrade to higher-quality methods: fuel-based > distance-based > spend-based",
                regulation_reference="GHG Protocol Scope 3, Chapter 7 - Data quality"
            )

        return None

    # ========================================================================
    # Additional Compliance Checks
    # ========================================================================

    def _check_uncertainty_quantification(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check uncertainty quantification."""
        if result.uncertainty_percentage is None:
            return ComplianceIssue(
                rule_id="UNCERTAINTY_QUANTIFICATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Uncertainty not quantified",
                recommendation="Quantify uncertainty using GHG Protocol Chapter 7 guidance (e.g., ±20% for spend-based, ±10% for distance-based)",
                regulation_reference="GHG Protocol Scope 3, Chapter 7.3"
            )

        return None

    def _check_reporting_period(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check reporting period definition."""
        if not result.reporting_period_start or not result.reporting_period_end:
            return ComplianceIssue(
                rule_id="REPORTING_PERIOD",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Reporting period not defined",
                recommendation="Define reporting period start and end dates",
                regulation_reference="GHG Protocol Corporate Standard, Chapter 3"
            )

        return None

    def _check_base_year(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check base year definition."""
        if not result.base_year:
            return ComplianceIssue(
                rule_id="BASE_YEAR",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Base year not defined",
                recommendation="Define base year for trend tracking and target setting",
                regulation_reference="GHG Protocol Corporate Standard, Chapter 5"
            )

        return None

    def _check_emission_breakdown(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check emission gas breakdown."""
        if result.co2_kg == 0 and result.ch4_kg == 0 and result.n2o_kg == 0:
            return ComplianceIssue(
                rule_id="EMISSION_BREAKDOWN",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="No gas-specific emission breakdown provided",
                recommendation="Report emissions by gas type (CO2, CH4, N2O) for transparency",
                regulation_reference="GHG Protocol Corporate Standard, Chapter 8"
            )

        return None

    def _check_tce_identification(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check Transport Chain Element (TCE) identification."""
        if not result.metadata.get("tce_identified"):
            return ComplianceIssue(
                rule_id="TCE_IDENTIFICATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Transport Chain Elements (TCEs) not explicitly identified",
                recommendation="Identify and document all Transport Chain Elements per ISO 14083",
                regulation_reference="ISO 14083:2023, Section 5.2"
            )

        return None

    def _check_default_values_documentation(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check default values documentation."""
        if not result.metadata.get("default_values_documented"):
            return ComplianceIssue(
                rule_id="DEFAULT_VALUES_DOCUMENTATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Use of default values not documented",
                recommendation="Document all default values used (load factors, distances, emission factors) and their sources",
                regulation_reference="ISO 14083:2023, Section 6.4"
            )

        return None

    def _check_multimodal_handling(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check multimodal transport handling."""
        if TransportMode.MULTIMODAL in result.transport_modes:
            if not result.metadata.get("multimodal_legs_separated"):
                return ComplianceIssue(
                    rule_id="MULTIMODAL_HANDLING",
                    severity=ComplianceSeverity.MEDIUM,
                    status=ComplianceStatus.WARNING,
                    message="Multimodal transport not separated into individual legs",
                    recommendation="Separate multimodal transport into individual mode-specific legs for accuracy",
                    regulation_reference="ISO 14083:2023, Multimodal transport"
                )

        return None

    def _check_empty_running_consideration(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check empty running consideration."""
        if not result.metadata.get("empty_running_considered"):
            return ComplianceIssue(
                rule_id="EMPTY_RUNNING",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Empty running not considered in calculations",
                recommendation="Consider empty running (empty return trips) in emission calculations or document why excluded",
                regulation_reference="ISO 14083:2023, Section 6.3.5"
            )

        return None

    def _check_load_factor_documentation(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check load factor documentation."""
        if not result.metadata.get("load_factor_documented"):
            return ComplianceIssue(
                rule_id="LOAD_FACTOR_DOCUMENTATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Load factors not documented",
                recommendation="Document load factors used (mass or volume utilization) for all transport modes",
                regulation_reference="ISO 14083:2023, Section 6.3.6"
            )

        return None

    def _check_energy_carrier_specification(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check energy carrier specification."""
        if not result.metadata.get("energy_carrier_specified"):
            return ComplianceIssue(
                rule_id="ENERGY_CARRIER_SPECIFICATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Energy carriers (fuels) not specified",
                recommendation="Specify energy carriers (diesel, gasoline, HFO, electricity, etc.) for each transport mode",
                regulation_reference="ISO 14083:2023, Section 6.2"
            )

        return None

    def _check_emission_factor_sources(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check emission factor sources."""
        if not result.metadata.get("ef_sources_documented"):
            return ComplianceIssue(
                rule_id="EMISSION_FACTOR_SOURCES",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Emission factor sources not documented",
                recommendation="Document all emission factor sources (GLEC, DEFRA, EPA, etc.) and versions",
                regulation_reference="GHG Protocol Scope 3, Chapter 7"
            )

        return None

    def _check_glec_emission_factors(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check GLEC-accredited emission factors."""
        if not result.metadata.get("glec_factors_used"):
            return ComplianceIssue(
                rule_id="GLEC_EMISSION_FACTORS",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="GLEC-accredited emission factors not used",
                recommendation="Use GLEC-accredited emission factors for logistics reporting",
                regulation_reference="GLEC Framework v3.2"
            )

        return None

    def _check_hub_to_hub_methodology(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check hub-to-hub methodology."""
        if not result.metadata.get("hub_to_hub_used"):
            return ComplianceIssue(
                rule_id="HUB_TO_HUB_METHODOLOGY",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Hub-to-hub methodology not used",
                recommendation="Use hub-to-hub methodology for multi-leg transport chains per GLEC Framework",
                regulation_reference="GLEC Framework v3.2, Section 4"
            )

        return None

    def _check_shipment_level_tracking(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check shipment-level tracking."""
        if not result.shipments:
            return ComplianceIssue(
                rule_id="SHIPMENT_LEVEL_TRACKING",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="No shipment-level data available",
                recommendation="Track emissions at shipment level for highest accuracy",
                regulation_reference="GLEC Framework v3.2"
            )

        return None

    def _check_tce_aggregation(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check TCE aggregation."""
        if not result.metadata.get("tce_aggregated"):
            return ComplianceIssue(
                rule_id="TCE_AGGREGATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Transport Chain Elements not aggregated",
                recommendation="Aggregate TCE emissions to shipment/route level per GLEC Framework",
                regulation_reference="GLEC Framework v3.2"
            )

        return None

    def _check_scope3_disclosure(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check Scope 3 disclosure."""
        if result.total_emissions_kg_co2e == 0:
            return ComplianceIssue(
                rule_id="SCOPE3_DISCLOSURE",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message="No Scope 3 Category 4 emissions reported",
                recommendation="Report Scope 3 Category 4 emissions or document exclusion rationale",
                regulation_reference="CSRD/ESRS E1-6"
            )

        return None

    def _check_double_materiality(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double materiality assessment."""
        if not result.metadata.get("double_materiality_assessed"):
            return ComplianceIssue(
                rule_id="DOUBLE_MATERIALITY",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Double materiality not assessed for upstream transportation",
                recommendation="Assess both impact materiality and financial materiality per ESRS",
                regulation_reference="CSRD/ESRS 1, Double materiality"
            )

        return None

    def _check_value_chain_mapping(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check value chain mapping."""
        if not result.metadata.get("value_chain_mapped"):
            return ComplianceIssue(
                rule_id="VALUE_CHAIN_MAPPING",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Value chain not mapped for upstream transportation",
                recommendation="Map value chain to identify all upstream transport activities",
                regulation_reference="CSRD/ESRS E1"
            )

        return None

    def _check_data_quality_disclosure(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check data quality disclosure."""
        if result.data_quality_score is None:
            return ComplianceIssue(
                rule_id="DATA_QUALITY_DISCLOSURE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Data quality not disclosed",
                recommendation="Disclose data quality using GHG Protocol's data quality indicators",
                regulation_reference="CSRD/ESRS E1, Data quality"
            )

        return None

    def _check_methodology_disclosure(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check methodology disclosure."""
        if not result.metadata.get("methodology_disclosed"):
            return ComplianceIssue(
                rule_id="METHODOLOGY_DISCLOSURE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Calculation methodology not disclosed",
                recommendation="Disclose calculation methodology, assumptions, and emission factors used",
                regulation_reference="CDP C6.5, CSRD/ESRS E1"
            )

        return None

    def _check_target_alignment(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check target alignment."""
        if not result.metadata.get("target_defined"):
            return ComplianceIssue(
                rule_id="TARGET_ALIGNMENT",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="No reduction target defined for upstream transportation",
                recommendation="Define reduction target aligned with corporate climate goals",
                regulation_reference="CSRD/ESRS E1, Climate targets"
            )

        return None

    def _check_boundary_definition(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check boundary definition."""
        if not result.metadata.get("boundary_defined"):
            return ComplianceIssue(
                rule_id="BOUNDARY_DEFINITION",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Organizational and operational boundaries not defined",
                recommendation="Define boundaries per GHG Protocol: equity share, operational control, or financial control",
                regulation_reference="GHG Protocol Corporate Standard, Chapter 3"
            )

        return None

    def _check_calculation_vs_estimation(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check calculation vs estimation percentage."""
        if not result.metadata.get("calculation_percentage"):
            return ComplianceIssue(
                rule_id="CALCULATION_VS_ESTIMATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Percentage of emissions calculated vs estimated not disclosed",
                recommendation="Disclose what percentage of emissions are based on primary data vs estimates",
                regulation_reference="CDP C6.5"
            )

        return None

    def _check_scope3_coverage(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check Scope 3 coverage for SBTi."""
        if not result.metadata.get("scope3_coverage_percentage"):
            return ComplianceIssue(
                rule_id="SCOPE3_COVERAGE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Scope 3 coverage percentage not documented",
                recommendation="Document Scope 3 coverage percentage (SBTi requires 67% if Scope 3 > 40% of total)",
                regulation_reference="SBTi Corporate Manual v2.0"
            )

        coverage = Decimal(str(result.metadata["scope3_coverage_percentage"]))
        if coverage < self.config.sbti_scope3_coverage_threshold:
            return ComplianceIssue(
                rule_id="SCOPE3_COVERAGE",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message=f"Scope 3 coverage is {coverage}%, below SBTi requirement of {self.config.sbti_scope3_coverage_threshold}%",
                details={"coverage_percentage": float(coverage)},
                recommendation="Expand Scope 3 coverage to at least 67% of total Scope 3 emissions",
                regulation_reference="SBTi Corporate Manual v2.0"
            )

        return None

    def _check_category4_in_target(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check Category 4 inclusion in target boundary."""
        if not result.metadata.get("cat4_in_target_boundary"):
            return ComplianceIssue(
                rule_id="CATEGORY4_IN_TARGET",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Category 4 inclusion in target boundary not documented",
                recommendation="Document whether Category 4 is included in Scope 3 reduction target",
                regulation_reference="SBTi Corporate Manual v2.0"
            )

        return None

    def _check_gri_305_3_disclosure(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check GRI 305-3 disclosure."""
        if result.total_emissions_kg_co2e == 0:
            return ComplianceIssue(
                rule_id="GRI_305_3_DISCLOSURE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="No GRI 305-3 (Scope 3) emissions reported",
                recommendation="Report Scope 3 emissions per GRI 305-3 or explain omission",
                regulation_reference="GRI 305: Emissions 2016"
            )

        return None

    def _check_emission_sources(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check emission sources identification."""
        if not result.transport_modes:
            return ComplianceIssue(
                rule_id="EMISSION_SOURCES",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Emission sources (transport modes) not identified",
                recommendation="Identify and disclose all emission sources (road, rail, sea, air, etc.)",
                regulation_reference="GRI 305-3-c"
            )

        return None

    def _check_consolidation_approach(
        self,
        result: CalculationResultInput
    ) -> Optional[ComplianceIssue]:
        """Check consolidation approach."""
        if not result.metadata.get("consolidation_approach"):
            return ComplianceIssue(
                rule_id="CONSOLIDATION_APPROACH",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Consolidation approach not disclosed",
                recommendation="Disclose consolidation approach (equity share, operational control, or financial control)",
                regulation_reference="GRI 305-3-d"
            )

        return None

    # ========================================================================
    # Scoring and Status Helpers
    # ========================================================================

    def _calculate_score(
        self,
        passed: int,
        failed: int,
        warnings: int,
        issues: List[ComplianceIssue]
    ) -> Decimal:
        """
        Calculate compliance score based on pass/fail/warning counts and issue severity.

        Scoring approach:
        - Start with base score from pass ratio
        - Apply weighted deductions for issues based on severity
        """
        total = passed + failed + warnings
        if total == 0:
            return Decimal("100")

        # Base score from pass ratio
        base_score = (Decimal(str(passed)) / Decimal(str(total))) * 100

        # Calculate weighted deductions
        deduction = Decimal("0")
        for issue in issues:
            if issue.severity == ComplianceSeverity.CRITICAL:
                deduction += self.config.critical_weight * 10
            elif issue.severity == ComplianceSeverity.HIGH:
                deduction += self.config.high_weight * 10
            elif issue.severity == ComplianceSeverity.MEDIUM:
                deduction += self.config.medium_weight * 10
            elif issue.severity == ComplianceSeverity.LOW:
                deduction += self.config.low_weight * 10

        final_score = max(Decimal("0"), base_score - deduction)

        return final_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _determine_status(
        self,
        score: Decimal,
        failed_count: int
    ) -> ComplianceStatus:
        """Determine overall compliance status."""
        if failed_count > 0:
            return ComplianceStatus.FAIL

        if score >= 95:
            return ComplianceStatus.PASS
        elif score >= 70:
            return ComplianceStatus.WARNING
        else:
            return ComplianceStatus.FAIL


# ============================================================================
# Module-level convenience functions
# ============================================================================


def create_compliance_checker(
    config: Optional[ComplianceCheckerConfig] = None
) -> ComplianceCheckerEngine:
    """
    Create a ComplianceCheckerEngine instance.

    Args:
        config: Optional configuration; uses defaults if not provided

    Returns:
        Configured ComplianceCheckerEngine instance

    Example:
        >>> checker = create_compliance_checker()
        >>> result = checker.check_all_frameworks(calc_result)
    """
    if config is None:
        config = ComplianceCheckerConfig()

    return ComplianceCheckerEngine(config)
