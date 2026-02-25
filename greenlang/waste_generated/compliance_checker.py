"""
ComplianceCheckerEngine - AGENT-MRV-018 Engine 6

This module implements the ComplianceCheckerEngine for Waste Generated in Operations
(GHG Protocol Scope 3 Category 5). It validates calculations against 7 regulatory frameworks
with Category 5-specific compliance rules.

Regulatory Frameworks:
1. GHG Protocol Scope 3 (Category 5 specific)
2. ISO 14064-1:2018 (Clause 5.2.4)
3. CSRD/ESRS E1 + E5
4. CDP Climate Change (Module 7)
5. SBTi (Category 5 materiality)
6. EU Waste Framework Directive
7. EPA 40 CFR Part 98 (Subpart HH/TT)

Category 5-Specific Compliance Rules:
- Third-party waste treatment boundary enforcement (exclude on-site → Scope 1/2)
- 4 calculation methods hierarchy validation
- Double-counting prevention (Cat 1, 12, Scope 1/2)
- Waste hierarchy compliance
- Diversion rate targets (EU 55%→60%→65%)
- Treatment method disclosure
- Recycling credit allocation (cut-off approach)
- Waste-to-energy vs Scope 2 boundary
- Transport of waste vs Category 4 boundary
- Hazardous waste separate reporting
- Biogenic vs fossil carbon separation
- FOD model parameter validation

Example:
    >>> engine = ComplianceCheckerEngine(config)
    >>> result = engine.check_all_frameworks(calculation_result)
    >>> overall_score = engine.get_overall_compliance_score(result)
    >>> print(f"Compliance: {overall_score}%")
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import logging
from threading import Lock

from pydantic import BaseModel, Field

from greenlang.waste_generated.models import (
    ComplianceFramework,
    CalculationMethod,
    WasteTreatmentMethod,
    WasteCategory,
    WasteStream,
    LandfillType,
    GasCollectionSystem,
    DataQualityTier,
    EFSource,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


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


class DoubleCountingCategory(str, Enum):
    """Scope 3 categories that could overlap with Category 5."""

    CATEGORY_1 = "CATEGORY_1"  # Purchased goods (cradle-to-gate vs cradle-to-grave)
    CATEGORY_12 = "CATEGORY_12"  # End-of-life of sold products
    SCOPE_1 = "SCOPE_1"  # On-site waste treatment
    SCOPE_2 = "SCOPE_2"  # Waste-to-energy purchased electricity
    CATEGORY_4 = "CATEGORY_4"  # Transport of waste to treatment facility


class WasteHierarchyLevel(str, Enum):
    """EU Waste Framework Directive hierarchy levels."""

    PREVENTION = "PREVENTION"  # Most preferred
    REUSE = "REUSE"
    RECYCLING = "RECYCLING"
    RECOVERY = "RECOVERY"  # Energy recovery
    DISPOSAL = "DISPOSAL"  # Least preferred (landfill)


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


class WasteResultInput(BaseModel):
    """Input calculation result for compliance checking."""

    # Total emissions
    total_emissions_kg_co2e: Decimal = Field(..., description="Total emissions")
    co2_fossil_kg: Decimal = Field(default=Decimal("0"), description="Fossil CO2")
    co2_biogenic_kg: Decimal = Field(default=Decimal("0"), description="Biogenic CO2")
    ch4_kg: Decimal = Field(default=Decimal("0"), description="CH4 emissions")
    n2o_kg: Decimal = Field(default=Decimal("0"), description="N2O emissions")

    # Calculation method
    calculation_method: CalculationMethod = Field(..., description="Method used")

    # Waste streams
    waste_streams: List[Dict[str, Any]] = Field(default_factory=list)
    total_waste_tonnes: Decimal = Field(default=Decimal("0"), description="Total waste mass")

    # Treatment breakdown
    treatment_distribution: Dict[WasteTreatmentMethod, Decimal] = Field(
        default_factory=dict,
        description="Waste mass by treatment method (tonnes)"
    )

    # Waste category breakdown
    category_distribution: Dict[WasteCategory, Decimal] = Field(
        default_factory=dict,
        description="Waste mass by category (tonnes)"
    )

    # Hazardous waste
    hazardous_waste_tonnes: Decimal = Field(default=Decimal("0"), description="Hazardous waste")
    non_hazardous_waste_tonnes: Decimal = Field(default=Decimal("0"), description="Non-hazardous")

    # Diversion metrics
    diverted_waste_tonnes: Decimal = Field(default=Decimal("0"), description="Diverted from landfill")
    landfilled_waste_tonnes: Decimal = Field(default=Decimal("0"), description="Landfilled waste")

    # Data quality
    data_quality_tier: Optional[DataQualityTier] = None
    spend_based_percentage: Optional[Decimal] = Field(None, ge=0, le=100)

    # Emission factor sources
    ef_sources: List[EFSource] = Field(default_factory=list)

    # Boundaries
    includes_transport_to_facility: bool = Field(default=False)
    includes_on_site_treatment: bool = Field(default=False)
    includes_waste_to_energy_credits: bool = Field(default=False)

    # Reporting
    reporting_period_start: Optional[datetime] = None
    reporting_period_end: Optional[datetime] = None
    base_year: Optional[int] = None
    reporting_year: Optional[int] = None

    # Uncertainty
    uncertainty_percentage: Optional[Decimal] = None

    # Scope 3 context (for materiality)
    total_scope3_emissions_kg_co2e: Optional[Decimal] = None
    category_5_percentage_of_scope3: Optional[Decimal] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceCheckerConfig(BaseModel):
    """Configuration for ComplianceCheckerEngine."""

    # Framework enablement
    enabled_frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: list(ComplianceFramework)
    )

    # Thresholds
    minimum_data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_2,
        description="Minimum acceptable data quality tier"
    )
    maximum_spend_based_percentage: Decimal = Field(
        default=Decimal("30.0"),
        ge=0,
        le=100,
        description="Max % of emissions from spend-based method (GHG Protocol: <30%)"
    )

    # EU Waste Directive targets
    eu_recycling_target_2025: Decimal = Field(default=Decimal("55.0"))
    eu_recycling_target_2030: Decimal = Field(default=Decimal("60.0"))
    eu_recycling_target_2035: Decimal = Field(default=Decimal("65.0"))

    # SBTi-specific
    sbti_category_5_materiality_threshold: Decimal = Field(
        default=Decimal("2.5"),
        ge=0,
        le=100,
        description="SBTi materiality threshold for Category 5 (% of Scope 3)"
    )
    sbti_scope3_coverage_threshold: Decimal = Field(
        default=Decimal("67.0"),
        ge=0,
        le=100,
        description="SBTi requires 67% Scope 3 coverage"
    )

    # EPA 40 CFR Part 98 thresholds
    epa_landfill_reporting_threshold_mtco2e: Decimal = Field(
        default=Decimal("25000"),
        description="EPA landfill reporting threshold (25,000 MTCO2e)"
    )

    # Tolerances
    allow_on_site_treatment: bool = Field(
        default=False,
        description="Allow on-site treatment (should be Scope 1/2)"
    )
    require_biogenic_separation: bool = Field(
        default=True,
        description="Require biogenic/fossil CO2 separation"
    )
    require_hazardous_separation: bool = Field(
        default=True,
        description="Require hazardous/non-hazardous separation"
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

    This engine implements comprehensive compliance checking for Waste Generated in
    Operations (Category 5) against major regulatory frameworks including GHG Protocol,
    ISO 14064, CSRD/ESRS E1+E5, CDP, SBTi, EU Waste Directive, and EPA 40 CFR Part 98.

    Category 5-Specific Compliance Rules:
    1. Third-party treatment boundary (exclude on-site → Scope 1/2)
    2. 4 calculation methods hierarchy validation
    3. Double-counting prevention (Cat 1, 12, Scope 1/2, Cat 4)
    4. Waste hierarchy compliance (EU Waste Directive)
    5. Diversion rate targets (EU: 55%/60%/65%)
    6. Treatment method disclosure
    7. Recycling credit allocation (cut-off approach)
    8. Waste-to-energy vs Scope 2 boundary
    9. Transport of waste vs Category 4
    10. Hazardous waste separate reporting
    11. Biogenic vs fossil carbon separation
    12. FOD model parameter validation
    13. Data quality tier assessment
    14. Materiality threshold (SBTi: 2.5%)
    15. EPA landfill reporting (25,000 MTCO2e threshold)

    Thread-Safe: Singleton pattern with lock for concurrent access.

    Attributes:
        config: Engine configuration
        _instance: Singleton instance
        _lock: Thread lock for singleton pattern

    Example:
        >>> config = ComplianceCheckerConfig()
        >>> engine = ComplianceCheckerEngine.get_instance(config)
        >>> result_input = WasteResultInput(...)
        >>> all_results = engine.check_all_frameworks(result_input)
        >>> overall_score = engine.get_overall_compliance_score(all_results)
    """

    _instance: Optional["ComplianceCheckerEngine"] = None
    _lock: Lock = Lock()

    def __init__(self, config: ComplianceCheckerConfig):
        """Initialize ComplianceCheckerEngine."""
        self.config = config

        # Waste hierarchy mapping (treatment method → hierarchy level)
        self.waste_hierarchy_map: Dict[WasteTreatmentMethod, WasteHierarchyLevel] = {
            WasteTreatmentMethod.RECYCLING_OPEN_LOOP: WasteHierarchyLevel.RECYCLING,
            WasteTreatmentMethod.RECYCLING_CLOSED_LOOP: WasteHierarchyLevel.RECYCLING,
            WasteTreatmentMethod.COMPOSTING: WasteHierarchyLevel.RECYCLING,
            WasteTreatmentMethod.ANAEROBIC_DIGESTION: WasteHierarchyLevel.RECOVERY,
            WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY: WasteHierarchyLevel.RECOVERY,
            WasteTreatmentMethod.LANDFILL_WITH_ENERGY_RECOVERY: WasteHierarchyLevel.RECOVERY,
            WasteTreatmentMethod.INCINERATION: WasteHierarchyLevel.DISPOSAL,
            WasteTreatmentMethod.LANDFILL: WasteHierarchyLevel.DISPOSAL,
            WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE: WasteHierarchyLevel.DISPOSAL,
            WasteTreatmentMethod.WASTEWATER_TREATMENT: WasteHierarchyLevel.DISPOSAL,
            WasteTreatmentMethod.OTHER: WasteHierarchyLevel.DISPOSAL,
        }

        # Diversion methods (exclude from landfill)
        self.diversion_methods: Set[WasteTreatmentMethod] = {
            WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            WasteTreatmentMethod.COMPOSTING,
            WasteTreatmentMethod.ANAEROBIC_DIGESTION,
            WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
        }

        logger.info(f"ComplianceCheckerEngine initialized with {len(config.enabled_frameworks)} frameworks")

    @classmethod
    def get_instance(cls, config: Optional[ComplianceCheckerConfig] = None) -> "ComplianceCheckerEngine":
        """
        Get singleton instance of ComplianceCheckerEngine (thread-safe).

        Args:
            config: Configuration (required for first call)

        Returns:
            Singleton instance

        Example:
            >>> engine = ComplianceCheckerEngine.get_instance(config)
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if config is None:
                        config = ComplianceCheckerConfig()
                    cls._instance = cls(config)
        return cls._instance

    # ========================================================================
    # Main Entry Points
    # ========================================================================

    def check_all_frameworks(
        self,
        result: WasteResultInput
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
                elif framework == ComplianceFramework.ISO_14064:
                    check_result = self.check_iso_14064(result)
                elif framework == ComplianceFramework.CSRD_ESRS:
                    check_result = self.check_csrd_esrs(result)
                elif framework == ComplianceFramework.CDP:
                    check_result = self.check_cdp(result)
                elif framework == ComplianceFramework.SBTI:
                    check_result = self.check_sbti(result)
                elif framework == ComplianceFramework.EU_WASTE_DIRECTIVE:
                    check_result = self.check_eu_waste_directive(result)
                elif framework == ComplianceFramework.EPA_40CFR98:
                    check_result = self.check_epa_40cfr98(result)
                else:
                    logger.warning(f"Unknown framework: {framework}")
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
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 (Category 5 specific).

        GHG Protocol Category 5 Requirements:
        1. Third-party treatment only (exclude on-site → Scope 1/2)
        2. 4 calculation methods hierarchy (supplier > waste-type > average > spend)
        3. Double-counting prevention (Cat 1, 12, Scope 1/2, Cat 4)
        4. Total waste generated disclosure
        5. Treatment method disclosure
        6. Emission factor sources documentation
        7. Allocation method disclosure (if multi-facility)
        8. Data quality assessment
        9. Uncertainty quantification
        10. Reporting period definition
        11. Base year consistency
        12. Waste-to-energy credit handling (cut-off approach)
        13. Transport to facility boundary (optional, avoid Cat 4 overlap)
        14. Recycling credit allocation
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

        # 1. Third-party treatment boundary
        issue = self._check_third_party_boundary(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 2. Calculation method hierarchy
        issue = self._check_calculation_method_hierarchy(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 3. Double-counting: Category 1 (purchased goods)
        issue = self._check_double_counting_cat1(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Double-counting: Category 12 (end-of-life)
        issue = self._check_double_counting_cat12(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Double-counting: Scope 1 (on-site treatment)
        issue = self._check_double_counting_scope1(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 6. Double-counting: Scope 2 (waste-to-energy)
        issue = self._check_double_counting_scope2(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Double-counting: Category 4 (transport to facility)
        issue = self._check_double_counting_cat4(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 8. Total waste generated disclosure
        issue = self._check_total_waste_disclosure(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 9. Treatment method disclosure
        issue = self._check_treatment_method_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Emission factor sources
        issue = self._check_ef_sources(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 11. Data quality assessment
        issue = self._check_data_quality(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 12. Spend-based percentage limit
        issue = self._check_spend_based_limit(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 13. Uncertainty quantification
        issue = self._check_uncertainty_quantification(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 14. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 15. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 3:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.GHG_PROTOCOL,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "category": "5",
                "category_name": "Waste Generated in Operations"
            }
        )

    # ========================================================================
    # ISO 14064-1 Compliance
    # ========================================================================

    def check_iso_14064(
        self,
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018 (Clause 5.2.4).

        ISO 14064-1 Requirements (Clause 5.2.4 - Indirect GHG emissions):
        1. Methodology documentation
        2. Emission factor sources and vintage
        3. Data quality assessment
        4. Uncertainty assessment (quantitative)
        5. Boundary definition (operational/organizational)
        6. Double-counting prevention
        7. Biogenic carbon separate reporting
        8. Quantification approach documentation
        9. Reporting period definition
        10. Base year definition and recalculation triggers

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for ISO 14064

        Example:
            >>> iso_result = engine.check_iso_14064(calc_result)
            >>> print(iso_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Methodology documentation
        issue = self._check_methodology_documentation(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 2. Emission factor sources
        issue = self._check_ef_sources_vintage(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Data quality assessment
        issue = self._check_data_quality(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Uncertainty assessment (quantitative)
        issue = self._check_uncertainty_quantitative(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 5. Boundary definition
        issue = self._check_boundary_definition(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Double-counting prevention
        issue = self._check_double_counting_all(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 7. Biogenic carbon separation
        issue = self._check_biogenic_separation(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 8. Quantification approach
        issue = self._check_quantification_approach(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 9. Reporting period
        issue = self._check_reporting_period(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 10. Base year
        issue = self._check_base_year(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 2:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.ISO_14064,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "standard": "ISO 14064-1:2018",
                "clause": "5.2.4"
            }
        )

    # ========================================================================
    # CSRD/ESRS E1 + E5 Compliance
    # ========================================================================

    def check_csrd_esrs(
        self,
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with CSRD/ESRS E1 + E5.

        CSRD/ESRS Requirements:
        - E1-6: Scope 3 Category 5 emissions (gross, location-based, market-based)
        - E5-1: Policies on circular economy
        - E5-2: Actions on circular economy
        - E5-3: Targets related to waste (diversion, recycling)
        - E5-4: Resource inflows (total waste generated)
        - E5-5: Resource outflows (waste by treatment method, hazardous breakdown)

        Specific Disclosures:
        1. Total waste generated (tonnes) by waste stream
        2. Waste diverted from disposal (tonnes and %)
        3. Waste directed to disposal (tonnes and %)
        4. Hazardous waste breakdown
        5. Treatment method breakdown (reuse, recycling, composting, incineration, landfill)
        6. Circular economy metrics (recycling rate, diversion rate)
        7. Scope 3 Category 5 emissions (GHG Protocol alignment)

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for CSRD/ESRS

        Example:
            >>> csrd_result = engine.check_csrd_esrs(calc_result)
            >>> print(csrd_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # E5-4: Total waste generated
        issue = self._check_total_waste_disclosure(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # E5-5: Waste diverted from disposal
        issue = self._check_waste_diversion_disclosure(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # E5-5: Waste directed to disposal
        issue = self._check_waste_disposal_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # E5-5: Hazardous waste breakdown
        issue = self._check_hazardous_waste_disclosure(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # E5-5: Treatment method breakdown
        issue = self._check_treatment_method_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # E5-5: Circular economy metrics
        issue = self._check_circular_economy_metrics(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # E1-6: Scope 3 Category 5 emissions
        issue = self._check_scope3_cat5_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # E1-6: GHG Protocol alignment
        issue = self._check_ghg_protocol_alignment(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 2:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.CSRD_ESRS,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "standards": ["ESRS E1-6", "ESRS E5-4", "ESRS E5-5"]
            }
        )

    # ========================================================================
    # CDP Climate Change Compliance
    # ========================================================================

    def check_cdp(
        self,
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change Questionnaire.

        CDP Requirements (Module 7 - Scope 3):
        1. Category 5 emissions disclosure
        2. Calculation methodology
        3. Emission factor sources
        4. Data quality assessment
        5. Verification/assurance status
        6. Percentage of emissions calculated vs estimated
        7. Upstream/downstream split
        8. Explanation of exclusions (if any)

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

        # 1. Category 5 emissions disclosure
        issue = self._check_scope3_cat5_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Calculation methodology
        issue = self._check_methodology_documentation(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 3. Emission factor sources
        issue = self._check_ef_sources(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Data quality assessment
        issue = self._check_data_quality(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Verification/assurance readiness
        issue = self._check_verification_readiness(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 6. Calculated vs estimated percentage
        issue = self._check_calculated_vs_estimated(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 2:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.CDP,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "questionnaire": "Climate Change",
                "module": "7 - Scope 3"
            }
        )

    # ========================================================================
    # SBTi Compliance
    # ========================================================================

    def check_sbti(
        self,
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative (SBTi).

        SBTi Requirements:
        1. Category 5 materiality assessment (>2.5% of Scope 3)
        2. 67% Scope 3 coverage requirement (if Cat 5 included)
        3. 2.5% annual reduction target (if Cat 5 in scope)
        4. GHG Protocol alignment
        5. Base year establishment
        6. Target setting (absolute or intensity)
        7. Boundary consistency

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for SBTi

        Example:
            >>> sbti_result = engine.check_sbti(calc_result)
            >>> print(sbti_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Materiality assessment
        issue = self._check_sbti_materiality(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 2. GHG Protocol alignment
        issue = self._check_ghg_protocol_alignment(result)
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

        # 4. Boundary consistency
        issue = self._check_boundary_definition(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 5. Data quality (for target tracking)
        issue = self._check_data_quality(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 1:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.SBTI,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "category_5_materiality_threshold": float(self.config.sbti_category_5_materiality_threshold),
                "scope3_coverage_threshold": float(self.config.sbti_scope3_coverage_threshold)
            }
        )

    # ========================================================================
    # EU Waste Framework Directive Compliance
    # ========================================================================

    def check_eu_waste_directive(
        self,
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with EU Waste Framework Directive (2008/98/EC).

        EU Waste Directive Requirements:
        1. Waste hierarchy compliance (prevention > reuse > recycling > recovery > disposal)
        2. Recycling targets:
           - 55% by 2025
           - 60% by 2030
           - 65% by 2035
        3. Separate collection requirements
        4. Extended Producer Responsibility (EPR) compliance
        5. Hazardous waste separate tracking
        6. Waste prevention measures
        7. Treatment method disclosure

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for EU Waste Directive

        Example:
            >>> eu_result = engine.check_eu_waste_directive(calc_result)
            >>> print(eu_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Waste hierarchy compliance
        issue = self._check_waste_hierarchy(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 2. Recycling targets (year-dependent)
        issue = self._check_recycling_targets(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 3. Diversion rate
        issue = self._check_diversion_rate(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # 4. Hazardous waste separation
        issue = self._check_hazardous_separation(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            else:
                warnings += 1
        else:
            passed += 1

        # 5. Treatment method disclosure
        issue = self._check_treatment_method_disclosure(result)
        if issue:
            issues.append(issue)
            warnings += 1
        else:
            passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 2:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.EU_WASTE_DIRECTIVE,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "directive": "2008/98/EC",
                "recycling_targets": {
                    "2025": float(self.config.eu_recycling_target_2025),
                    "2030": float(self.config.eu_recycling_target_2030),
                    "2035": float(self.config.eu_recycling_target_2035)
                }
            }
        )

    # ========================================================================
    # EPA 40 CFR Part 98 Compliance
    # ========================================================================

    def check_epa_40cfr98(
        self,
        result: WasteResultInput
    ) -> ComplianceCheckResult:
        """
        Check compliance with EPA 40 CFR Part 98 (Subpart HH/TT).

        EPA 40 CFR Part 98 Requirements:
        - Subpart HH: Municipal Solid Waste Landfills
          - Reporting threshold: 25,000 MTCO2e/year
          - FOD model (Equations HH-1 through HH-8)
          - Gas collection system data
          - Electronic reporting via e-GGRT
        - Subpart TT: Industrial Waste Landfills
          - Reporting threshold: 25,000 MTCO2e/year
          - Similar FOD model requirements

        Args:
            result: Calculation result to validate

        Returns:
            Compliance check result for EPA 40 CFR Part 98

        Example:
            >>> epa_result = engine.check_epa_40cfr98(calc_result)
            >>> print(epa_result.score)
        """
        issues = []
        passed = 0
        failed = 0
        warnings = 0

        # 1. Reporting threshold check
        issue = self._check_epa_reporting_threshold(result)
        if issue:
            issues.append(issue)
            if issue.status == ComplianceStatus.FAIL:
                failed += 1
            elif issue.status == ComplianceStatus.WARNING:
                warnings += 1
            # INFO status doesn't count as pass/fail/warning
        else:
            passed += 1

        # 2. FOD model compliance (if threshold exceeded)
        if result.total_emissions_kg_co2e >= self.config.epa_landfill_reporting_threshold_mtco2e * 1000:
            issue = self._check_fod_model_compliance(result)
            if issue:
                issues.append(issue)
                if issue.status == ComplianceStatus.FAIL:
                    failed += 1
                else:
                    warnings += 1
            else:
                passed += 1

            # 3. Gas collection system data
            issue = self._check_gas_collection_data(result)
            if issue:
                issues.append(issue)
                warnings += 1
            else:
                passed += 1

            # 4. Landfill categorization (MSW vs industrial)
            issue = self._check_landfill_categorization(result)
            if issue:
                issues.append(issue)
                warnings += 1
            else:
                passed += 1

        # Calculate score
        total_checks = passed + failed + warnings
        score = self._calculate_score(passed, failed, warnings)

        # Determine overall status
        if failed > 0:
            status = ComplianceStatus.FAIL
        elif warnings > 1:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.PASS

        return ComplianceCheckResult(
            framework=ComplianceFramework.EPA_40CFR98,
            status=status,
            score=score,
            issues=issues,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            total_checks=total_checks,
            metadata={
                "regulation": "40 CFR Part 98",
                "subparts": ["HH", "TT"],
                "reporting_threshold_mtco2e": float(self.config.epa_landfill_reporting_threshold_mtco2e)
            }
        )

    # ========================================================================
    # Double-Counting Prevention
    # ========================================================================

    def check_double_counting(
        self,
        result: WasteResultInput,
        other_category_results: Dict[DoubleCountingCategory, Dict[str, Any]]
    ) -> List[str]:
        """
        Check for double-counting across Scope 3 categories and Scope 1/2.

        Double-Counting Scenarios:
        1. Category 1 (Purchased Goods): Cradle-to-gate vs cradle-to-grave EF
           - If Cat 1 uses cradle-to-grave EF → waste already included
        2. Category 12 (End-of-Life): Operational waste vs sold product waste
           - Cat 5 = waste from operations
           - Cat 12 = waste from sold products
        3. Scope 1: On-site waste treatment (combustion, wastewater)
           - If on-site treatment → Scope 1, not Cat 5
        4. Scope 2: Waste-to-energy purchased electricity
           - If facility purchases electricity from WTE plant → Scope 2
           - If facility's waste goes to WTE → Cat 5 (with optional credit)
        5. Category 4: Transport of waste to treatment facility
           - If transport included in Cat 4 → exclude from Cat 5

        Args:
            result: Calculation result to validate
            other_category_results: Results from other categories/scopes

        Returns:
            List of double-counting warnings

        Example:
            >>> warnings = engine.check_double_counting(result, other_results)
            >>> for warning in warnings:
            ...     print(f"WARNING: {warning}")
        """
        warnings = []

        # Check Category 1 overlap
        if DoubleCountingCategory.CATEGORY_1 in other_category_results:
            cat1_data = other_category_results[DoubleCountingCategory.CATEGORY_1]
            if cat1_data.get("uses_cradle_to_grave_ef", False):
                warnings.append(
                    "Category 1 uses cradle-to-grave emission factors which may include "
                    "waste disposal. Verify that waste emissions are not double-counted "
                    "between Category 1 and Category 5."
                )

        # Check Category 12 overlap
        if DoubleCountingCategory.CATEGORY_12 in other_category_results:
            warnings.append(
                "Category 12 (End-of-Life of Sold Products) is active. Ensure Category 5 "
                "only includes operational waste, not waste from sold products."
            )

        # Check Scope 1 overlap
        if result.includes_on_site_treatment:
            warnings.append(
                "CRITICAL: On-site waste treatment detected in Category 5. On-site waste "
                "treatment (combustion, wastewater) should be reported in Scope 1, not Scope 3."
            )

        # Check Scope 2 overlap
        if result.includes_waste_to_energy_credits:
            if DoubleCountingCategory.SCOPE_2 in other_category_results:
                scope2_data = other_category_results[DoubleCountingCategory.SCOPE_2]
                if scope2_data.get("includes_wte_electricity", False):
                    warnings.append(
                        "Waste-to-energy credits detected in Category 5, and Scope 2 includes "
                        "WTE electricity. Ensure electricity from WTE is not double-counted."
                    )

        # Check Category 4 overlap
        if result.includes_transport_to_facility:
            if DoubleCountingCategory.CATEGORY_4 in other_category_results:
                warnings.append(
                    "Transport to waste treatment facility included in Category 5. "
                    "Verify this transport is not also included in Category 4 (Upstream Transportation)."
                )

        return warnings

    def check_materiality(
        self,
        category5_total: Decimal,
        scope3_total: Decimal
    ) -> Dict[str, Any]:
        """
        Check if Category 5 is material per SBTi criteria.

        SBTi Materiality Criteria:
        - Category is material if >2.5% of total Scope 3 emissions
        - If material → must be included in SBTi target
        - If not material → can be excluded (but must justify)

        Args:
            category5_total: Category 5 emissions (kg CO2e)
            scope3_total: Total Scope 3 emissions (kg CO2e)

        Returns:
            Dictionary with materiality assessment

        Example:
            >>> assessment = engine.check_materiality(cat5_total, scope3_total)
            >>> if assessment["is_material"]:
            ...     print("Category 5 must be included in SBTi target")
        """
        if scope3_total == 0:
            return {
                "is_material": False,
                "percentage_of_scope3": Decimal("0"),
                "threshold": self.config.sbti_category_5_materiality_threshold,
                "status": "UNKNOWN",
                "message": "Cannot assess materiality: Scope 3 total is zero"
            }

        percentage = (category5_total / scope3_total * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        is_material = percentage >= self.config.sbti_category_5_materiality_threshold

        return {
            "is_material": is_material,
            "percentage_of_scope3": float(percentage),
            "threshold": float(self.config.sbti_category_5_materiality_threshold),
            "status": "MATERIAL" if is_material else "NOT_MATERIAL",
            "message": (
                f"Category 5 represents {percentage}% of Scope 3 emissions "
                f"(threshold: {self.config.sbti_category_5_materiality_threshold}%)"
            )
        }

    def calculate_diversion_rate(
        self,
        result: WasteResultInput
    ) -> Decimal:
        """
        Calculate waste diversion rate (% diverted from landfill).

        Diversion Rate = (Diverted Waste / Total Waste) × 100

        Diverted Waste includes:
        - Recycling (open-loop and closed-loop)
        - Composting
        - Anaerobic digestion
        - Incineration with energy recovery

        Args:
            result: Calculation result with waste data

        Returns:
            Diversion rate (0-100%)

        Example:
            >>> rate = engine.calculate_diversion_rate(result)
            >>> print(f"Diversion rate: {rate}%")
        """
        if result.total_waste_tonnes == 0:
            return Decimal("0")

        diverted_tonnes = Decimal("0")
        for treatment, tonnes in result.treatment_distribution.items():
            if treatment in self.diversion_methods:
                diverted_tonnes += tonnes

        diversion_rate = (diverted_tonnes / result.total_waste_tonnes * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return diversion_rate

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_required_disclosures(
        self,
        framework: ComplianceFramework
    ) -> List[str]:
        """
        Get list of required disclosures for a framework.

        Args:
            framework: Regulatory framework

        Returns:
            List of required disclosure items

        Example:
            >>> disclosures = engine.get_required_disclosures(ComplianceFramework.GHG_PROTOCOL)
            >>> for disclosure in disclosures:
            ...     print(f"- {disclosure}")
        """
        disclosures = {
            ComplianceFramework.GHG_PROTOCOL: [
                "Total waste generated (tonnes)",
                "Waste by treatment method (tonnes and %)",
                "Category 5 emissions (kg CO2e)",
                "Calculation method used",
                "Emission factor sources",
                "Data quality assessment",
                "Uncertainty quantification",
                "Double-counting prevention measures",
                "Boundary definition (third-party only)",
                "Reporting period",
                "Base year",
            ],
            ComplianceFramework.ISO_14064: [
                "Methodology documentation",
                "Emission factor sources and vintage",
                "Data quality assessment",
                "Quantitative uncertainty assessment",
                "Boundary definition",
                "Biogenic carbon separate reporting",
                "Reporting period",
                "Base year and recalculation policy",
            ],
            ComplianceFramework.CSRD_ESRS: [
                "Total waste generated (E5-4)",
                "Waste diverted from disposal (E5-5)",
                "Waste directed to disposal (E5-5)",
                "Hazardous waste breakdown (E5-5)",
                "Treatment method breakdown (E5-5)",
                "Recycling rate (%)",
                "Diversion rate (%)",
                "Scope 3 Category 5 emissions (E1-6)",
            ],
            ComplianceFramework.CDP: [
                "Category 5 emissions (kg CO2e)",
                "Calculation methodology",
                "Emission factor sources",
                "Data quality assessment",
                "Verification/assurance status",
                "Percentage calculated vs estimated",
            ],
            ComplianceFramework.SBTI: [
                "Materiality assessment (% of Scope 3)",
                "GHG Protocol alignment",
                "Base year emissions",
                "Target year and reduction ambition",
                "Boundary consistency",
            ],
            ComplianceFramework.EU_WASTE_DIRECTIVE: [
                "Total waste generated (tonnes)",
                "Recycling rate (%)",
                "Diversion rate (%)",
                "Hazardous waste (tonnes)",
                "Treatment method breakdown",
                "Waste hierarchy compliance",
            ],
            ComplianceFramework.EPA_40CFR98: [
                "Total landfill emissions (MTCO2e)",
                "FOD model parameters (MCF, DOC, k)",
                "Gas collection system type and efficiency",
                "Landfill categorization (MSW vs industrial)",
                "Electronic reporting via e-GGRT",
            ],
        }

        return disclosures.get(framework, [])

    def check_data_completeness(
        self,
        result: WasteResultInput
    ) -> Dict[str, Any]:
        """
        Check data completeness for compliance reporting.

        Args:
            result: Calculation result to validate

        Returns:
            Dictionary with completeness assessment

        Example:
            >>> completeness = engine.check_data_completeness(result)
            >>> print(f"Completeness: {completeness['percentage']}%")
        """
        required_fields = [
            "total_emissions_kg_co2e",
            "total_waste_tonnes",
            "calculation_method",
            "treatment_distribution",
            "ef_sources",
            "reporting_period_start",
            "reporting_period_end",
        ]

        optional_fields = [
            "base_year",
            "data_quality_tier",
            "uncertainty_percentage",
            "total_scope3_emissions_kg_co2e",
        ]

        present_required = 0
        present_optional = 0

        # Check required fields
        for field in required_fields:
            value = getattr(result, field, None)
            if value is not None:
                if isinstance(value, (list, dict)):
                    if len(value) > 0:
                        present_required += 1
                elif isinstance(value, Decimal):
                    if value > 0:
                        present_required += 1
                else:
                    present_required += 1

        # Check optional fields
        for field in optional_fields:
            value = getattr(result, field, None)
            if value is not None:
                if isinstance(value, Decimal):
                    if value > 0:
                        present_optional += 1
                else:
                    present_optional += 1

        total_fields = len(required_fields) + len(optional_fields)
        total_present = present_required + present_optional
        completeness_percentage = (Decimal(total_present) / Decimal(total_fields) * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return {
            "percentage": float(completeness_percentage),
            "required_present": present_required,
            "required_total": len(required_fields),
            "optional_present": present_optional,
            "optional_total": len(optional_fields),
            "missing_required": [
                f for f in required_fields
                if not getattr(result, f, None)
            ],
            "missing_optional": [
                f for f in optional_fields
                if not getattr(result, f, None)
            ],
        }

    def generate_compliance_report(
        self,
        result: WasteResultInput,
        frameworks: List[ComplianceFramework]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report across frameworks.

        Args:
            result: Calculation result to validate
            frameworks: List of frameworks to check

        Returns:
            Compliance report dictionary

        Example:
            >>> report = engine.generate_compliance_report(result, [
            ...     ComplianceFramework.GHG_PROTOCOL,
            ...     ComplianceFramework.CSRD_ESRS
            ... ])
            >>> print(report["overall_compliance_score"])
        """
        # Run all framework checks
        self.config.enabled_frameworks = frameworks
        framework_results = self.check_all_frameworks(result)

        # Get overall metrics
        overall_score = self.get_overall_compliance_score(framework_results)
        summary = self.get_compliance_summary(framework_results)

        # Calculate diversion rate
        diversion_rate = self.calculate_diversion_rate(result)

        # Check data completeness
        completeness = self.check_data_completeness(result)

        # Materiality assessment (if Scope 3 total provided)
        materiality = None
        if result.total_scope3_emissions_kg_co2e:
            materiality = self.check_materiality(
                result.total_emissions_kg_co2e,
                result.total_scope3_emissions_kg_co2e
            )

        return {
            "report_generated_at": datetime.utcnow().isoformat(),
            "overall_compliance_score": float(overall_score),
            "overall_status": summary["overall_status"],
            "frameworks_checked": frameworks,
            "framework_results": {
                framework.value: {
                    "score": float(check_result.score),
                    "status": check_result.status.value,
                    "passed": check_result.passed_checks,
                    "failed": check_result.failed_checks,
                    "warnings": check_result.warning_checks,
                    "issues": [
                        {
                            "rule_id": issue.rule_id,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "recommendation": issue.recommendation
                        }
                        for issue in check_result.issues
                    ]
                }
                for framework, check_result in framework_results.items()
            },
            "waste_metrics": {
                "total_waste_tonnes": float(result.total_waste_tonnes),
                "diversion_rate_percentage": float(diversion_rate),
                "hazardous_waste_tonnes": float(result.hazardous_waste_tonnes),
                "treatment_distribution": {
                    method.value: float(tonnes)
                    for method, tonnes in result.treatment_distribution.items()
                }
            },
            "emissions_summary": {
                "total_kg_co2e": float(result.total_emissions_kg_co2e),
                "co2_fossil_kg": float(result.co2_fossil_kg),
                "co2_biogenic_kg": float(result.co2_biogenic_kg),
                "ch4_kg": float(result.ch4_kg),
                "n2o_kg": float(result.n2o_kg),
            },
            "materiality": materiality,
            "data_completeness": completeness,
            "recommendations": summary["recommendations"],
            "critical_issues": summary["critical_issues"],
            "high_issues": summary["high_issues"],
        }

    def validate_ef_sources(
        self,
        result: WasteResultInput
    ) -> List[str]:
        """
        Validate emission factor sources.

        Args:
            result: Calculation result to validate

        Returns:
            List of validation warnings

        Example:
            >>> warnings = engine.validate_ef_sources(result)
            >>> for warning in warnings:
            ...     print(f"WARNING: {warning}")
        """
        warnings = []

        if not result.ef_sources:
            warnings.append("No emission factor sources documented")
            return warnings

        # Check for custom EF (requires additional documentation)
        if EFSource.CUSTOM in result.ef_sources:
            warnings.append(
                "Custom emission factors detected. Ensure third-party verification "
                "and methodology documentation are available."
            )

        # Recommend specific sources
        if EFSource.EPA_WARM not in result.ef_sources and EFSource.DEFRA_BEIS not in result.ef_sources:
            warnings.append(
                "Consider using EPA WARM or DEFRA/BEIS emission factors for waste, "
                "as they are widely accepted and regularly updated."
            )

        return warnings

    def validate_boundary(
        self,
        result: WasteResultInput
    ) -> List[str]:
        """
        Validate organizational/operational boundary.

        Args:
            result: Calculation result to validate

        Returns:
            List of boundary validation warnings

        Example:
            >>> warnings = engine.validate_boundary(result)
            >>> for warning in warnings:
            ...     print(f"WARNING: {warning}")
        """
        warnings = []

        # Check for on-site treatment (should be Scope 1/2)
        if result.includes_on_site_treatment:
            warnings.append(
                "CRITICAL: On-site waste treatment should be reported in Scope 1 (direct emissions) "
                "or Scope 2 (electricity for treatment), not Scope 3 Category 5."
            )

        # Check for transport to facility boundary
        if result.includes_transport_to_facility:
            warnings.append(
                "Transport to waste treatment facility is included. Ensure this is not "
                "double-counted with Category 4 (Upstream Transportation)."
            )

        # Check for waste-to-energy credits
        if result.includes_waste_to_energy_credits:
            warnings.append(
                "Waste-to-energy credits included. Ensure electricity from WTE is not "
                "double-counted in Scope 2 purchased electricity."
            )

        return warnings

    # ========================================================================
    # Internal Check Methods
    # ========================================================================

    def _check_third_party_boundary(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check third-party treatment boundary (exclude on-site)."""
        if result.includes_on_site_treatment:
            return ComplianceIssue(
                rule_id="GHG_CAT5_BOUNDARY",
                severity=ComplianceSeverity.CRITICAL,
                status=ComplianceStatus.FAIL,
                message="On-site waste treatment detected in Category 5",
                details={"includes_on_site_treatment": True},
                recommendation=(
                    "On-site waste treatment (incineration, wastewater treatment) should be "
                    "reported in Scope 1 or Scope 2, not Scope 3 Category 5. Category 5 is "
                    "for third-party waste treatment only."
                ),
                regulation_reference="GHG Protocol Scope 3 Standard, Chapter 7"
            )
        return None

    def _check_calculation_method_hierarchy(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check calculation method follows hierarchy."""
        # GHG Protocol hierarchy: supplier-specific > waste-type-specific > average-data > spend-based
        if result.calculation_method == CalculationMethod.SPEND_BASED:
            if result.spend_based_percentage and result.spend_based_percentage > self.config.maximum_spend_based_percentage:
                return ComplianceIssue(
                    rule_id="GHG_CAT5_METHOD_HIERARCHY",
                    severity=ComplianceSeverity.HIGH,
                    status=ComplianceStatus.WARNING,
                    message=f"Spend-based method exceeds recommended maximum ({result.spend_based_percentage}%)",
                    details={
                        "calculation_method": result.calculation_method.value,
                        "spend_based_percentage": float(result.spend_based_percentage),
                        "maximum_allowed": float(self.config.maximum_spend_based_percentage)
                    },
                    recommendation=(
                        f"Reduce spend-based calculation to <{self.config.maximum_spend_based_percentage}% "
                        "by collecting waste-type-specific or supplier-specific data."
                    ),
                    regulation_reference="GHG Protocol Technical Guidance for Category 5, Table 7.1"
                )
        return None

    def _check_double_counting_cat1(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double-counting with Category 1."""
        # This is a warning to check if Cat 1 uses cradle-to-grave EF
        return ComplianceIssue(
            rule_id="GHG_CAT5_DOUBLE_COUNT_CAT1",
            severity=ComplianceSeverity.MEDIUM,
            status=ComplianceStatus.WARNING,
            message="Verify no double-counting with Category 1 (Purchased Goods)",
            details={"category": "1"},
            recommendation=(
                "If Category 1 uses cradle-to-grave emission factors (including disposal), "
                "ensure those same waste streams are not also counted in Category 5."
            ),
            regulation_reference="GHG Protocol Scope 3 Standard, Chapter 7.5"
        )

    def _check_double_counting_cat12(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double-counting with Category 12."""
        return ComplianceIssue(
            rule_id="GHG_CAT5_DOUBLE_COUNT_CAT12",
            severity=ComplianceSeverity.MEDIUM,
            status=ComplianceStatus.WARNING,
            message="Verify no double-counting with Category 12 (End-of-Life)",
            details={"category": "12"},
            recommendation=(
                "Category 5 = operational waste. Category 12 = waste from sold products. "
                "Ensure clear boundary between the two categories."
            ),
            regulation_reference="GHG Protocol Scope 3 Standard, Chapter 7.5"
        )

    def _check_double_counting_scope1(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double-counting with Scope 1."""
        if result.includes_on_site_treatment:
            return ComplianceIssue(
                rule_id="GHG_CAT5_DOUBLE_COUNT_SCOPE1",
                severity=ComplianceSeverity.CRITICAL,
                status=ComplianceStatus.FAIL,
                message="On-site waste treatment must be in Scope 1, not Category 5",
                details={"includes_on_site_treatment": True},
                recommendation=(
                    "Move on-site waste treatment emissions to Scope 1. "
                    "Only third-party waste treatment belongs in Category 5."
                ),
                regulation_reference="GHG Protocol Corporate Standard, Chapter 4"
            )
        return None

    def _check_double_counting_scope2(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double-counting with Scope 2."""
        if result.includes_waste_to_energy_credits:
            return ComplianceIssue(
                rule_id="GHG_CAT5_DOUBLE_COUNT_SCOPE2",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Waste-to-energy credits detected",
                details={"includes_waste_to_energy_credits": True},
                recommendation=(
                    "If your facility purchases electricity from a WTE plant, ensure that "
                    "electricity is not double-counted in both Scope 2 and Category 5 credits."
                ),
                regulation_reference="GHG Protocol Scope 3 Standard, Chapter 7.5"
            )
        return None

    def _check_double_counting_cat4(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double-counting with Category 4."""
        if result.includes_transport_to_facility:
            return ComplianceIssue(
                rule_id="GHG_CAT5_DOUBLE_COUNT_CAT4",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Transport to waste facility included in Category 5",
                details={"includes_transport_to_facility": True},
                recommendation=(
                    "Verify that transport of waste to treatment facility is not also "
                    "included in Category 4 (Upstream Transportation)."
                ),
                regulation_reference="GHG Protocol Scope 3 Standard, Chapter 7.5"
            )
        return None

    def _check_total_waste_disclosure(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check total waste generated disclosure."""
        if result.total_waste_tonnes <= 0:
            return ComplianceIssue(
                rule_id="GHG_CAT5_TOTAL_WASTE",
                severity=ComplianceSeverity.CRITICAL,
                status=ComplianceStatus.FAIL,
                message="Total waste generated not disclosed or zero",
                details={"total_waste_tonnes": float(result.total_waste_tonnes)},
                recommendation="Disclose total waste generated in tonnes.",
                regulation_reference="GHG Protocol Scope 3 Standard, Table 7.1"
            )
        return None

    def _check_treatment_method_disclosure(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check treatment method disclosure."""
        if not result.treatment_distribution:
            return ComplianceIssue(
                rule_id="GHG_CAT5_TREATMENT_METHODS",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message="Treatment method breakdown not provided",
                details={"treatment_distribution": {}},
                recommendation=(
                    "Disclose waste mass by treatment method (landfill, incineration, "
                    "recycling, composting, etc.)."
                ),
                regulation_reference="GHG Protocol Scope 3 Standard, Table 7.1"
            )
        return None

    def _check_ef_sources(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check emission factor sources."""
        if not result.ef_sources:
            return ComplianceIssue(
                rule_id="GHG_CAT5_EF_SOURCES",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Emission factor sources not documented",
                details={"ef_sources": []},
                recommendation=(
                    "Document emission factor sources (e.g., EPA WARM, DEFRA/BEIS, "
                    "IPCC 2006, custom)."
                ),
                regulation_reference="GHG Protocol Scope 3 Standard, Chapter 7.3"
            )
        return None

    def _check_data_quality(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check data quality tier."""
        if result.data_quality_tier is None:
            return ComplianceIssue(
                rule_id="GHG_CAT5_DATA_QUALITY",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Data quality tier not assessed",
                details={"data_quality_tier": None},
                recommendation="Assess data quality tier (Tier 1/2/3 per IPCC guidance).",
                regulation_reference="IPCC 2006 Guidelines, Volume 1 Chapter 4"
            )
        elif result.data_quality_tier == DataQualityTier.TIER_1:
            return ComplianceIssue(
                rule_id="GHG_CAT5_DATA_QUALITY",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Data quality is Tier 1 (default factors)",
                details={"data_quality_tier": result.data_quality_tier.value},
                recommendation="Improve to Tier 2/3 with country-specific or facility-specific data.",
                regulation_reference="IPCC 2006 Guidelines, Volume 1 Chapter 4"
            )
        return None

    def _check_spend_based_limit(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check spend-based percentage limit."""
        if result.spend_based_percentage and result.spend_based_percentage > self.config.maximum_spend_based_percentage:
            return ComplianceIssue(
                rule_id="GHG_CAT5_SPEND_BASED_LIMIT",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message=f"Spend-based method exceeds {self.config.maximum_spend_based_percentage}%",
                details={
                    "spend_based_percentage": float(result.spend_based_percentage),
                    "maximum_allowed": float(self.config.maximum_spend_based_percentage)
                },
                recommendation=(
                    f"Reduce spend-based calculation to <{self.config.maximum_spend_based_percentage}% "
                    "by collecting waste mass and treatment data."
                ),
                regulation_reference="GHG Protocol Technical Guidance for Category 5"
            )
        return None

    def _check_uncertainty_quantification(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check uncertainty quantification."""
        if result.uncertainty_percentage is None:
            return ComplianceIssue(
                rule_id="GHG_CAT5_UNCERTAINTY",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Uncertainty not quantified",
                details={"uncertainty_percentage": None},
                recommendation="Quantify uncertainty using IPCC default ranges or Monte Carlo simulation.",
                regulation_reference="IPCC 2006 Guidelines, Volume 1 Chapter 3"
            )
        return None

    def _check_reporting_period(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check reporting period definition."""
        if result.reporting_period_start is None or result.reporting_period_end is None:
            return ComplianceIssue(
                rule_id="GHG_CAT5_REPORTING_PERIOD",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Reporting period not defined",
                details={
                    "reporting_period_start": None,
                    "reporting_period_end": None
                },
                recommendation="Define reporting period start and end dates.",
                regulation_reference="GHG Protocol Scope 3 Standard, Chapter 9"
            )
        return None

    def _check_base_year(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check base year definition."""
        if result.base_year is None:
            return ComplianceIssue(
                rule_id="GHG_CAT5_BASE_YEAR",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Base year not defined",
                details={"base_year": None},
                recommendation="Define base year for tracking emissions over time.",
                regulation_reference="GHG Protocol Scope 3 Standard, Chapter 9"
            )
        return None

    def _check_methodology_documentation(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check methodology documentation."""
        if not result.metadata.get("methodology_documented", False):
            return ComplianceIssue(
                rule_id="ISO_14064_METHODOLOGY",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.WARNING,
                message="Methodology not documented",
                details={"methodology_documented": False},
                recommendation=(
                    "Document quantification methodology including calculation approach, "
                    "data sources, emission factors, and assumptions."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4"
            )
        return None

    def _check_ef_sources_vintage(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check emission factor sources and vintage."""
        if not result.metadata.get("ef_vintage", None):
            return ComplianceIssue(
                rule_id="ISO_14064_EF_VINTAGE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Emission factor vintage not documented",
                details={"ef_vintage": None},
                recommendation="Document emission factor vintage (publication year).",
                regulation_reference="ISO 14064-1:2018, Clause 5.2.4"
            )
        return None

    def _check_uncertainty_quantitative(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check quantitative uncertainty assessment."""
        if result.uncertainty_percentage is None:
            return ComplianceIssue(
                rule_id="ISO_14064_UNCERTAINTY",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.FAIL,
                message="Quantitative uncertainty assessment required by ISO 14064",
                details={"uncertainty_percentage": None},
                recommendation=(
                    "Conduct quantitative uncertainty assessment using IPCC Tier 1/2 methods "
                    "or Monte Carlo simulation."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.7"
            )
        return None

    def _check_boundary_definition(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check boundary definition."""
        if not result.metadata.get("boundary_defined", False):
            return ComplianceIssue(
                rule_id="ISO_14064_BOUNDARY",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Organizational/operational boundary not defined",
                details={"boundary_defined": False},
                recommendation=(
                    "Define organizational boundary (equity share, financial control, "
                    "operational control) and operational boundary (third-party treatment only)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.3"
            )
        return None

    def _check_double_counting_all(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check double-counting prevention (all categories)."""
        if not result.metadata.get("double_counting_checked", False):
            return ComplianceIssue(
                rule_id="ISO_14064_DOUBLE_COUNTING",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Double-counting prevention not documented",
                details={"double_counting_checked": False},
                recommendation=(
                    "Document measures to prevent double-counting across Scope 1/2/3 "
                    "and between Scope 3 categories."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.5"
            )
        return None

    def _check_biogenic_separation(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check biogenic carbon separation."""
        if self.config.require_biogenic_separation:
            if result.co2_biogenic_kg == 0 and result.co2_fossil_kg > 0:
                # May be OK if no biogenic waste
                if any(cat in result.category_distribution for cat in [
                    WasteCategory.FOOD_WASTE,
                    WasteCategory.GARDEN_WASTE,
                    WasteCategory.PAPER_CARDBOARD,
                    WasteCategory.WOOD
                ]):
                    return ComplianceIssue(
                        rule_id="ISO_14064_BIOGENIC",
                        severity=ComplianceSeverity.HIGH,
                        status=ComplianceStatus.FAIL,
                        message="Biogenic CO2 separation required but not provided",
                        details={
                            "co2_biogenic_kg": float(result.co2_biogenic_kg),
                            "co2_fossil_kg": float(result.co2_fossil_kg)
                        },
                        recommendation=(
                            "Separately report biogenic and fossil CO2 emissions. "
                            "Food waste, garden waste, paper, and wood are biogenic."
                        ),
                        regulation_reference="ISO 14064-1:2018, Clause 5.2.6"
                    )
        return None

    def _check_quantification_approach(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check quantification approach documentation."""
        if not result.metadata.get("quantification_approach", None):
            return ComplianceIssue(
                rule_id="ISO_14064_QUANTIFICATION",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Quantification approach not documented",
                details={"quantification_approach": None},
                recommendation=(
                    "Document quantification approach (calculation-based, measurement-based, "
                    "or hybrid)."
                ),
                regulation_reference="ISO 14064-1:2018, Clause 5.2.2"
            )
        return None

    def _check_waste_diversion_disclosure(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check waste diversion disclosure (ESRS E5-5)."""
        if result.diverted_waste_tonnes <= 0:
            return ComplianceIssue(
                rule_id="CSRD_E5_5_DIVERSION",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.FAIL,
                message="ESRS E5-5 requires waste diverted from disposal disclosure",
                details={"diverted_waste_tonnes": float(result.diverted_waste_tonnes)},
                recommendation=(
                    "Disclose waste diverted from disposal (recycling, composting, "
                    "energy recovery) in tonnes and as % of total waste."
                ),
                regulation_reference="ESRS E5-5"
            )
        return None

    def _check_waste_disposal_disclosure(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check waste disposal disclosure (ESRS E5-5)."""
        if result.landfilled_waste_tonnes <= 0:
            return ComplianceIssue(
                rule_id="CSRD_E5_5_DISPOSAL",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="ESRS E5-5 requires waste directed to disposal disclosure",
                details={"landfilled_waste_tonnes": float(result.landfilled_waste_tonnes)},
                recommendation=(
                    "Disclose waste directed to disposal (landfill, incineration without "
                    "energy recovery) in tonnes and as % of total waste."
                ),
                regulation_reference="ESRS E5-5"
            )
        return None

    def _check_hazardous_waste_disclosure(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check hazardous waste disclosure (ESRS E5-5)."""
        if self.config.require_hazardous_separation:
            if result.hazardous_waste_tonnes == 0 and result.non_hazardous_waste_tonnes == 0:
                return ComplianceIssue(
                    rule_id="CSRD_E5_5_HAZARDOUS",
                    severity=ComplianceSeverity.HIGH,
                    status=ComplianceStatus.FAIL,
                    message="ESRS E5-5 requires hazardous/non-hazardous waste breakdown",
                    details={
                        "hazardous_waste_tonnes": float(result.hazardous_waste_tonnes),
                        "non_hazardous_waste_tonnes": float(result.non_hazardous_waste_tonnes)
                    },
                    recommendation=(
                        "Separately report hazardous and non-hazardous waste in tonnes."
                    ),
                    regulation_reference="ESRS E5-5"
                )
        return None

    def _check_circular_economy_metrics(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check circular economy metrics (ESRS E5)."""
        diversion_rate = self.calculate_diversion_rate(result)
        if diversion_rate < 50:
            return ComplianceIssue(
                rule_id="CSRD_E5_CIRCULAR",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message=f"Low diversion rate ({diversion_rate}%)",
                details={"diversion_rate": float(diversion_rate)},
                recommendation=(
                    "Improve waste diversion rate through increased recycling, composting, "
                    "and energy recovery. Target >50% diversion."
                ),
                regulation_reference="ESRS E5-3"
            )
        return None

    def _check_scope3_cat5_disclosure(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check Scope 3 Category 5 disclosure."""
        if result.total_emissions_kg_co2e <= 0:
            return ComplianceIssue(
                rule_id="CSRD_E1_6_CAT5",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="ESRS E1-6 requires Scope 3 Category 5 emissions disclosure",
                details={"total_emissions_kg_co2e": float(result.total_emissions_kg_co2e)},
                recommendation="Disclose Scope 3 Category 5 emissions in kg CO2e.",
                regulation_reference="ESRS E1-6"
            )
        return None

    def _check_ghg_protocol_alignment(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check GHG Protocol alignment."""
        # Verify calculation method is one of the 4 GHG Protocol methods
        valid_methods = [
            CalculationMethod.SUPPLIER_SPECIFIC,
            CalculationMethod.WASTE_TYPE_SPECIFIC,
            CalculationMethod.AVERAGE_DATA,
            CalculationMethod.SPEND_BASED
        ]
        if result.calculation_method not in valid_methods:
            return ComplianceIssue(
                rule_id="GHG_PROTOCOL_ALIGNMENT",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Calculation method not aligned with GHG Protocol",
                details={"calculation_method": result.calculation_method.value},
                recommendation=(
                    "Use one of the 4 GHG Protocol methods: supplier-specific, "
                    "waste-type-specific, average-data, or spend-based."
                ),
                regulation_reference="GHG Protocol Scope 3 Standard, Table 7.1"
            )
        return None

    def _check_verification_readiness(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check verification/assurance readiness."""
        if not result.metadata.get("verification_ready", False):
            return ComplianceIssue(
                rule_id="CDP_VERIFICATION",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Verification readiness not assessed",
                details={"verification_ready": False},
                recommendation=(
                    "Prepare for third-party verification by documenting methodology, "
                    "data sources, assumptions, and emission factors."
                ),
                regulation_reference="CDP Climate Change Questionnaire, C10"
            )
        return None

    def _check_calculated_vs_estimated(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check calculated vs estimated percentage."""
        if not result.metadata.get("calculated_percentage", None):
            return ComplianceIssue(
                rule_id="CDP_CALCULATED_VS_ESTIMATED",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Calculated vs estimated percentage not disclosed",
                details={"calculated_percentage": None},
                recommendation=(
                    "Disclose percentage of emissions calculated from primary data vs "
                    "estimated from secondary data or spend."
                ),
                regulation_reference="CDP Climate Change Questionnaire, C6.5"
            )
        return None

    def _check_sbti_materiality(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check SBTi materiality threshold."""
        if result.total_scope3_emissions_kg_co2e is None:
            return ComplianceIssue(
                rule_id="SBTI_MATERIALITY_UNKNOWN",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Cannot assess SBTi materiality without total Scope 3 emissions",
                details={"total_scope3_emissions_kg_co2e": None},
                recommendation=(
                    "Provide total Scope 3 emissions to assess Category 5 materiality "
                    f"(threshold: {self.config.sbti_category_5_materiality_threshold}%)."
                ),
                regulation_reference="SBTi Criteria, Version 5.2"
            )
        else:
            materiality = self.check_materiality(
                result.total_emissions_kg_co2e,
                result.total_scope3_emissions_kg_co2e
            )
            if materiality["is_material"]:
                return ComplianceIssue(
                    rule_id="SBTI_MATERIALITY",
                    severity=ComplianceSeverity.INFO,
                    status=ComplianceStatus.PASS,
                    message=f"Category 5 is material ({materiality['percentage_of_scope3']}% of Scope 3)",
                    details=materiality,
                    recommendation=(
                        "Category 5 must be included in SBTi target with 2.5% annual reduction."
                    ),
                    regulation_reference="SBTi Criteria, Version 5.2"
                )
        return None

    def _check_waste_hierarchy(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check waste hierarchy compliance."""
        # Calculate % of waste by hierarchy level
        hierarchy_distribution = {}
        for treatment, tonnes in result.treatment_distribution.items():
            level = self.waste_hierarchy_map.get(treatment, WasteHierarchyLevel.DISPOSAL)
            hierarchy_distribution[level] = hierarchy_distribution.get(level, Decimal("0")) + tonnes

        # Check if >50% is disposal (landfill/incineration)
        disposal_tonnes = hierarchy_distribution.get(WasteHierarchyLevel.DISPOSAL, Decimal("0"))
        disposal_percentage = (disposal_tonnes / result.total_waste_tonnes * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ) if result.total_waste_tonnes > 0 else Decimal("0")

        if disposal_percentage > 50:
            return ComplianceIssue(
                rule_id="EU_WASTE_HIERARCHY",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message=f"{disposal_percentage}% of waste directed to disposal (landfill/incineration)",
                details={
                    "disposal_percentage": float(disposal_percentage),
                    "hierarchy_distribution": {
                        level.value: float(tonnes)
                        for level, tonnes in hierarchy_distribution.items()
                    }
                },
                recommendation=(
                    "Follow EU waste hierarchy: prevention > reuse > recycling > recovery > disposal. "
                    "Increase recycling and recovery to reduce disposal."
                ),
                regulation_reference="EU Waste Framework Directive 2008/98/EC, Article 4"
            )
        return None

    def _check_recycling_targets(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check EU recycling targets (year-dependent)."""
        diversion_rate = self.calculate_diversion_rate(result)

        if result.reporting_year is None:
            return ComplianceIssue(
                rule_id="EU_RECYCLING_TARGET_YEAR_UNKNOWN",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message="Cannot assess recycling target without reporting year",
                details={"reporting_year": None},
                recommendation="Specify reporting year to assess EU recycling targets.",
                regulation_reference="EU Waste Framework Directive 2008/98/EC, Article 11"
            )

        # Determine target based on year
        if result.reporting_year >= 2035:
            target = self.config.eu_recycling_target_2035
            target_year = 2035
        elif result.reporting_year >= 2030:
            target = self.config.eu_recycling_target_2030
            target_year = 2030
        elif result.reporting_year >= 2025:
            target = self.config.eu_recycling_target_2025
            target_year = 2025
        else:
            # Pre-2025, no mandatory target
            return None

        if diversion_rate < target:
            return ComplianceIssue(
                rule_id="EU_RECYCLING_TARGET",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.FAIL,
                message=f"Diversion rate ({diversion_rate}%) below {target_year} target ({target}%)",
                details={
                    "diversion_rate": float(diversion_rate),
                    "target": float(target),
                    "target_year": target_year
                },
                recommendation=(
                    f"Increase diversion rate to meet {target_year} EU target of {target}% "
                    "through increased recycling, composting, and energy recovery."
                ),
                regulation_reference="EU Waste Framework Directive 2008/98/EC, Article 11"
            )
        return None

    def _check_diversion_rate(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check diversion rate."""
        diversion_rate = self.calculate_diversion_rate(result)
        if diversion_rate < 50:
            return ComplianceIssue(
                rule_id="EU_DIVERSION_RATE",
                severity=ComplianceSeverity.LOW,
                status=ComplianceStatus.WARNING,
                message=f"Low diversion rate ({diversion_rate}%)",
                details={"diversion_rate": float(diversion_rate)},
                recommendation=(
                    "Improve waste diversion through increased recycling, composting, "
                    "and energy recovery. Target >50% diversion."
                ),
                regulation_reference="EU Waste Framework Directive 2008/98/EC"
            )
        return None

    def _check_hazardous_separation(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check hazardous waste separation."""
        if self.config.require_hazardous_separation:
            if result.hazardous_waste_tonnes == 0 and result.non_hazardous_waste_tonnes == 0:
                return ComplianceIssue(
                    rule_id="EU_HAZARDOUS_SEPARATION",
                    severity=ComplianceSeverity.HIGH,
                    status=ComplianceStatus.FAIL,
                    message="Hazardous/non-hazardous waste separation required",
                    details={
                        "hazardous_waste_tonnes": float(result.hazardous_waste_tonnes),
                        "non_hazardous_waste_tonnes": float(result.non_hazardous_waste_tonnes)
                    },
                    recommendation=(
                        "Separately track and report hazardous and non-hazardous waste per "
                        "EU Waste Framework Directive."
                    ),
                    regulation_reference="EU Waste Framework Directive 2008/98/EC, Article 18"
                )
        return None

    def _check_epa_reporting_threshold(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check EPA 40 CFR Part 98 reporting threshold."""
        threshold_kg = self.config.epa_landfill_reporting_threshold_mtco2e * 1000

        if result.total_emissions_kg_co2e >= threshold_kg:
            return ComplianceIssue(
                rule_id="EPA_40CFR98_THRESHOLD",
                severity=ComplianceSeverity.INFO,
                status=ComplianceStatus.PASS,
                message=f"Emissions ({result.total_emissions_kg_co2e / 1000:.0f} MTCO2e) exceed EPA reporting threshold",
                details={
                    "total_emissions_mtco2e": float(result.total_emissions_kg_co2e / 1000),
                    "threshold_mtco2e": float(self.config.epa_landfill_reporting_threshold_mtco2e)
                },
                recommendation=(
                    "EPA 40 CFR Part 98 reporting required. Submit annual report via e-GGRT."
                ),
                regulation_reference="40 CFR Part 98 Subpart HH, §98.340"
            )
        else:
            return ComplianceIssue(
                rule_id="EPA_40CFR98_THRESHOLD",
                severity=ComplianceSeverity.INFO,
                status=ComplianceStatus.WARNING,
                message=f"Emissions ({result.total_emissions_kg_co2e / 1000:.0f} MTCO2e) below EPA reporting threshold",
                details={
                    "total_emissions_mtco2e": float(result.total_emissions_kg_co2e / 1000),
                    "threshold_mtco2e": float(self.config.epa_landfill_reporting_threshold_mtco2e)
                },
                recommendation=(
                    "EPA reporting not required (below 25,000 MTCO2e threshold)."
                ),
                regulation_reference="40 CFR Part 98 Subpart HH, §98.340"
            )

    def _check_fod_model_compliance(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check FOD model compliance (EPA Equations HH-1 through HH-8)."""
        if not result.metadata.get("fod_model_used", False):
            return ComplianceIssue(
                rule_id="EPA_40CFR98_FOD_MODEL",
                severity=ComplianceSeverity.HIGH,
                status=ComplianceStatus.FAIL,
                message="EPA requires FOD model for landfill emissions (Equations HH-1 to HH-8)",
                details={"fod_model_used": False},
                recommendation=(
                    "Use IPCC First Order Decay (FOD) model per EPA 40 CFR Part 98 Subpart HH. "
                    "Calculate CH4 generation using Equations HH-1 through HH-8."
                ),
                regulation_reference="40 CFR Part 98 Subpart HH, §98.343"
            )
        return None

    def _check_gas_collection_data(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check gas collection system data."""
        if not result.metadata.get("gas_collection_documented", False):
            return ComplianceIssue(
                rule_id="EPA_40CFR98_GAS_COLLECTION",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Gas collection system data not documented",
                details={"gas_collection_documented": False},
                recommendation=(
                    "Document gas collection system type, collection efficiency, "
                    "and flaring/energy recovery information."
                ),
                regulation_reference="40 CFR Part 98 Subpart HH, §98.344"
            )
        return None

    def _check_landfill_categorization(
        self,
        result: WasteResultInput
    ) -> Optional[ComplianceIssue]:
        """Check landfill categorization (MSW vs industrial)."""
        if not result.metadata.get("landfill_type", None):
            return ComplianceIssue(
                rule_id="EPA_40CFR98_LANDFILL_TYPE",
                severity=ComplianceSeverity.MEDIUM,
                status=ComplianceStatus.WARNING,
                message="Landfill type not categorized (MSW vs industrial)",
                details={"landfill_type": None},
                recommendation=(
                    "Categorize landfill as MSW (Subpart HH) or industrial (Subpart TT)."
                ),
                regulation_reference="40 CFR Part 98 Subpart HH/TT"
            )
        return None

    def _calculate_score(
        self,
        passed: int,
        failed: int,
        warnings: int
    ) -> Decimal:
        """
        Calculate compliance score (0-100).

        Scoring formula:
        - Passed: +1.0
        - Warning: +0.5
        - Failed: +0.0
        Score = (passed + 0.5*warnings) / total * 100

        Args:
            passed: Number of passed checks
            failed: Number of failed checks
            warnings: Number of warning checks

        Returns:
            Compliance score (0-100)
        """
        total = passed + failed + warnings
        if total == 0:
            return Decimal("100")

        weighted_score = (
            Decimal(passed) * self.config.critical_weight +
            Decimal(warnings) * self.config.medium_weight
        )

        score = (weighted_score / Decimal(total) * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return score
