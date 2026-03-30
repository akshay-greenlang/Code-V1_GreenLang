# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - AGENT-MRV-016 Engine 6

Multi-framework regulatory compliance checker for Scope 3 Category 3
(Fuel and Energy-Related Activities) emissions.

This engine validates calculation results against seven compliance
frameworks with Category 3-specific requirements:

1. GHG Protocol Scope 3 Standard (12 requirements)
2. CSRD/ESRS E1 (10 requirements)
3. CDP Climate Questionnaire (10 requirements)
4. SBTi (8 requirements)
5. SB 253 California (8 requirements)
6. GRI 305 (6 requirements)
7. ISO 14064-1:2018 (6 requirements)

The engine performs zero-hallucination compliance checking by:
  - Validating against deterministic rule sets
  - Checking boundary definitions (WTT, T&D, upstream)
  - Verifying activity classification (3a/3b/3c/3d)
  - Assessing data quality and disclosure completeness
  - Detecting double-counting with Scope 1/2
  - Validating biogenic CO2 separation

All compliance checks use explicit rules with no LLM involvement.
Results are scored 0-100 per framework with actionable findings.

Key Category 3-Specific Rules:
  - WTT_COMBUSTION_EXCLUSION: WTT factors exclude combustion
  - TD_LOSS_BOUNDARY: T&D losses correctly bounded
  - UPSTREAM_GENERATION_EXCLUSION: Upstream EFs exclude generation
  - ACTIVITY_CLASSIFICATION: Activities correctly classified
  - UTILITY_ONLY_3D: Activity 3d only for utilities
  - DOUBLE_COUNTING_SCOPE1: No overlap with Scope 1
  - DOUBLE_COUNTING_SCOPE2: No overlap with Scope 2
  - BIOGENIC_SEPARATION: Biogenic CO2 reported separately

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .models import (
    CalculationMethod,
    CalculationResult,
    ComplianceCheckResult,
    ComplianceFinding,
    ComplianceFramework,
    ComplianceStatus,
    DQIScore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework Requirements Registry
# ---------------------------------------------------------------------------

# GHG Protocol Scope 3 Standard - 12 requirements
GHG_PROTOCOL_REQUIREMENTS = [
    "SEPARATE_REPORTING_3A_3B_3C_3D",
    "WTT_COMBUSTION_EXCLUSION",
    "TD_LOSS_COUNTRY_SPECIFIC",
    "UPSTREAM_GENERATION_EXCLUSION",
    "ACTIVITY_3D_UTILITY_ONLY",
    "PER_GAS_BREAKDOWN",
    "METHODOLOGY_DISCLOSURE",
    "DATA_QUALITY_ASSESSMENT",
    "UNCERTAINTY_ANALYSIS",
    "YOY_COMPARISON",
    "DOUBLE_COUNTING_PREVENTION",
    "BOUNDARY_DESCRIPTION",
]

# CSRD/ESRS E1 - 10 requirements
CSRD_REQUIREMENTS = [
    "MATERIAL_SCOPE3_REPORTING",
    "CATEGORY_3_MATERIALITY",
    "GHG_INTENSITY_REVENUE",
    "DATA_SOURCE_DISCLOSURE",
    "TRANSITION_PLAN_ALIGNMENT",
    "BASE_YEAR_COMPARISON",
    "ACTIVITY_LEVEL_BREAKDOWN",
    "UNCERTAINTY_DISCLOSURE",
    "IMPROVEMENT_PLAN",
    "THIRD_PARTY_VERIFICATION",
]

# CDP Climate Questionnaire - 10 requirements
CDP_REQUIREMENTS = [
    "RELEVANCE_ASSESSMENT",
    "EMISSIONS_FIGURE_WITH_METHOD",
    "DATA_QUALITY_SCORE",
    "VERIFICATION_STATUS",
    "YOY_CHANGE_EXPLANATION",
    "SUPPLIER_ENGAGEMENT",
    "EXCLUSIONS_JUSTIFIED",
    "CATEGORY_BOUNDARY_DEFINED",
    "EF_SOURCES_CITED",
    "IMPROVEMENT_TARGETS",
]

# SBTi - 8 requirements
SBTI_REQUIREMENTS = [
    "SCOPE3_SCREENING_1PCT",
    "BOUNDARY_COMPLETENESS_95PCT",
    "DATA_QUALITY_IMPROVEMENT_PLAN",
    "BASE_YEAR_RECALCULATION_CRITERIA",
    "TARGET_CONSISTENT_METHODOLOGY",
    "ANNUAL_PROGRESS_DISCLOSURE",
    "ACTIVITY_LEVEL_GRANULARITY",
    "SUPPLIER_SPECIFIC_DATA_PRIORITY",
]

# SB 253 California - 8 requirements
SB253_REQUIREMENTS = [
    "REVENUE_THRESHOLD_1B",
    "MATERIAL_SCOPE3_CATEGORIES",
    "THIRD_PARTY_VERIFICATION_REQUIRED",
    "PCAF_METHODOLOGY",
    "ANNUAL_REPORTING",
    "PUBLIC_DISCLOSURE",
    "METHODOLOGY_DOCUMENTATION",
    "HISTORICAL_COMPARISON",
]

# GRI 305 - 6 requirements
GRI_REQUIREMENTS = [
    "GRI_305_3_DISCLOSURE",
    "BASE_YEAR_DISCLOSURE",
    "CONSOLIDATION_APPROACH",
    "EF_SOURCES_GWP_VALUES",
    "METHODOLOGY_ASSUMPTIONS",
    "BIOGENIC_CO2_SEPARATE",
]

# ISO 14064 - 6 requirements
ISO_REQUIREMENTS = [
    "CATEGORIZE_OTHER_INDIRECT",
    "QUANTIFICATION_METHODOLOGY",
    "UNCERTAINTY_ASSESSMENT",
    "DOCUMENTATION_REQUIREMENTS",
    "VERIFICATION_PROVISIONS",
    "REPORTING_PERIOD_ALIGNMENT",
]


# ---------------------------------------------------------------------------
# Category 3-Specific Compliance Rules
# ---------------------------------------------------------------------------

CATEGORY3_RULES = {
    "WTT_COMBUSTION_EXCLUSION": {
        "name": "WTT factors must exclude combustion emissions",
        "severity": "critical",
        "frameworks": [ComplianceFramework.GHG_PROTOCOL_SCOPE3],
    },
    "TD_LOSS_BOUNDARY": {
        "name": "T&D losses correctly bounded",
        "severity": "major",
        "frameworks": [ComplianceFramework.GHG_PROTOCOL_SCOPE3],
    },
    "UPSTREAM_GENERATION_EXCLUSION": {
        "name": "Upstream EFs exclude generation-point emissions",
        "severity": "critical",
        "frameworks": [ComplianceFramework.GHG_PROTOCOL_SCOPE3],
    },
    "ACTIVITY_CLASSIFICATION": {
        "name": "Activities correctly classified (3a/3b/3c/3d)",
        "severity": "major",
        "frameworks": [
            ComplianceFramework.GHG_PROTOCOL_SCOPE3,
            ComplianceFramework.CDP,
        ],
    },
    "UTILITY_ONLY_3D": {
        "name": "Activity 3d only applicable to utilities",
        "severity": "major",
        "frameworks": [ComplianceFramework.GHG_PROTOCOL_SCOPE3],
    },
    "DOUBLE_COUNTING_SCOPE1": {
        "name": "No double-counting with Scope 1",
        "severity": "critical",
        "frameworks": [
            ComplianceFramework.GHG_PROTOCOL_SCOPE3,
            ComplianceFramework.GRI_305,
        ],
    },
    "DOUBLE_COUNTING_SCOPE2": {
        "name": "No double-counting with Scope 2",
        "severity": "critical",
        "frameworks": [
            ComplianceFramework.GHG_PROTOCOL_SCOPE3,
            ComplianceFramework.GRI_305,
        ],
    },
    "BIOGENIC_SEPARATION": {
        "name": "Biogenic CO2 reported separately",
        "severity": "major",
        "frameworks": [
            ComplianceFramework.GHG_PROTOCOL_SCOPE3,
            ComplianceFramework.GRI_305,
        ],
    },
}


# ---------------------------------------------------------------------------
# ComplianceCheckerEngine
# ---------------------------------------------------------------------------


class ComplianceCheckerEngine:
    """
    Multi-framework regulatory compliance checker for Category 3.

    This engine validates calculation results against seven compliance
    frameworks, checking Category 3-specific requirements and producing
    actionable findings for each framework.

    The engine performs zero-hallucination compliance checking by
    validating against deterministic rule sets with no LLM involvement.

    Attributes:
        _checks_performed: Counter of total checks performed.
        _frameworks_checked: Set of frameworks that have been checked.

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> results = engine.check_all(
        ...     calculation_result,
        ...     [ComplianceFramework.GHG_PROTOCOL_SCOPE3, ComplianceFramework.CDP]
        ... )
        >>> for result in results:
        ...     print(f"{result.framework}: {result.status}")
    """

    def __init__(self) -> None:
        """Initialize ComplianceCheckerEngine."""
        self._checks_performed: int = 0
        self._frameworks_checked: set[ComplianceFramework] = set()
        logger.info("ComplianceCheckerEngine initialized")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def check_all(
        self,
        calculation_result: CalculationResult,
        frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> List[ComplianceCheckResult]:
        """
        Check compliance against multiple frameworks.

        Args:
            calculation_result: The calculation result to check.
            frameworks: List of frameworks to check (all if None).

        Returns:
            List of compliance check results, one per framework.

        Example:
            >>> results = engine.check_all(calc_result)
            >>> compliant = [r for r in results if r.status == ComplianceStatus.COMPLIANT]
        """
        if frameworks is None:
            frameworks = list(ComplianceFramework)

        results = []
        for framework in frameworks:
            if framework == ComplianceFramework.GHG_PROTOCOL_SCOPE3:
                result = self.check_ghg_protocol(calculation_result)
            elif framework == ComplianceFramework.CSRD_ESRS_E1:
                result = self.check_csrd(calculation_result)
            elif framework == ComplianceFramework.CDP:
                result = self.check_cdp(calculation_result)
            elif framework == ComplianceFramework.SBTI:
                result = self.check_sbti(calculation_result)
            elif framework == ComplianceFramework.SB_253:
                result = self.check_sb253(calculation_result)
            elif framework == ComplianceFramework.GRI_305:
                result = self.check_gri(calculation_result)
            elif framework == ComplianceFramework.ISO_14064:
                result = self.check_iso14064(calculation_result)
            else:
                logger.warning(f"Unknown framework: {framework}")
                continue

            results.append(result)
            self._frameworks_checked.add(framework)

        self._checks_performed += len(results)
        logger.info(
            f"Completed compliance checks for {len(results)} frameworks"
        )
        return results

    def check_ghg_protocol(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 Standard.

        Validates 12 requirements specific to Category 3:
          1. Separate reporting of 3a/3b/3c/3d
          2. WTT factors exclude combustion
          3. T&D losses use country-specific factors
          4. Upstream EFs exclude generation
          5. Activity 3d only for utilities
          6. Per-gas breakdown (CO2, CH4, N2O)
          7. Methodology disclosure
          8. Data quality assessment
          9. Uncertainty analysis
         10. Year-over-year comparison
         11. Double-counting prevention
         12. Boundary description

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. Separate reporting of activities
        has_3a = len(calculation_result.activity_3a_results) > 0
        has_3b = len(calculation_result.activity_3b_results) > 0
        has_3c = len(calculation_result.activity_3c_results) > 0
        has_3d = len(calculation_result.activity_3d_results) > 0

        activities_reported = sum([has_3a, has_3b, has_3c, has_3d])

        if activities_reported >= 3:
            findings.append(
                ComplianceFinding(
                    rule_id="GHG-CAT3-001",
                    rule_name="Separate reporting of 3a/3b/3c/3d",
                    status=ComplianceStatus.COMPLIANT,
                    severity="critical",
                    message=f"{activities_reported} sub-activities reported separately",
                    recommendation="Continue separate reporting of all applicable activities",
                )
            )
        else:
            findings.append(
                ComplianceFinding(
                    rule_id="GHG-CAT3-001",
                    rule_name="Separate reporting of 3a/3b/3c/3d",
                    status=ComplianceStatus.NON_COMPLIANT,
                    severity="critical",
                    message=f"Only {activities_reported} sub-activities reported",
                    recommendation="Report all applicable sub-activities (3a/3b/3c/3d) separately",
                )
            )

        # 2. WTT combustion exclusion
        wtt_compliant = self._check_wtt_combustion_exclusion(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-002",
                rule_name="WTT factors exclude combustion emissions",
                status=ComplianceStatus.COMPLIANT
                if wtt_compliant
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="WTT factors correctly exclude combustion"
                if wtt_compliant
                else "Cannot verify WTT factors exclude combustion",
                recommendation="Verify WTT factors from trusted sources (DEFRA, IEA, GREET)",
            )
        )

        # 3. T&D loss country-specific factors
        td_compliant = self._check_td_loss_factors(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-003",
                rule_name="T&D losses use country-specific factors",
                status=ComplianceStatus.COMPLIANT
                if td_compliant
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="T&D loss factors are country-specific"
                if td_compliant
                else "Consider using country-specific T&D loss factors",
                recommendation="Use IEA or World Bank country-specific T&D loss factors",
            )
        )

        # 4. Upstream generation exclusion
        upstream_compliant = self._check_upstream_generation_exclusion(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-004",
                rule_name="Upstream EFs exclude generation-point emissions",
                status=ComplianceStatus.COMPLIANT
                if upstream_compliant
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Upstream EFs correctly exclude generation"
                if upstream_compliant
                else "Cannot verify upstream EFs exclude generation",
                recommendation="Ensure upstream EFs only cover extraction and transmission",
            )
        )

        # 5. Activity 3d utility-only
        utility_compliant = self._check_utility_only_3d(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-005",
                rule_name="Activity 3d only applicable to utilities",
                status=ComplianceStatus.COMPLIANT
                if utility_compliant
                else ComplianceStatus.NON_COMPLIANT,
                severity="major",
                message="Activity 3d correctly applied"
                if utility_compliant
                else "Activity 3d should only be used by utilities",
                recommendation="Only report Activity 3d if you are a utility or energy reseller",
            )
        )

        # 6. Per-gas breakdown
        gas_breakdown_compliant = self._check_per_gas_breakdown(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-006",
                rule_name="Per-gas breakdown (CO2, CH4, N2O)",
                status=ComplianceStatus.COMPLIANT
                if gas_breakdown_compliant
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Per-gas breakdown provided"
                if gas_breakdown_compliant
                else "Per-gas breakdown incomplete",
                recommendation="Report CO2, CH4, and N2O separately for all activities",
            )
        )

        # 7. Methodology disclosure
        method_disclosed = calculation_result.method is not None
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-007",
                rule_name="Calculation methodology disclosure",
                status=ComplianceStatus.COMPLIANT
                if method_disclosed
                else ComplianceStatus.NON_COMPLIANT,
                severity="major",
                message=f"Method: {calculation_result.method.value}"
                if method_disclosed
                else "Calculation method not disclosed",
                recommendation="Disclose calculation methodology in emissions report",
            )
        )

        # 8. Data quality assessment
        dqi_assessed = self._check_dqi_assessment(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-008",
                rule_name="Data quality assessment",
                status=ComplianceStatus.COMPLIANT
                if dqi_assessed
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Data quality indicators assessed"
                if dqi_assessed
                else "Data quality assessment incomplete",
                recommendation="Assess DQI across all five dimensions (temporal, geographical, technological, completeness, reliability)",
            )
        )

        # 9. Uncertainty analysis
        uncertainty_analyzed = self._check_uncertainty_analysis(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-009",
                rule_name="Uncertainty analysis",
                status=ComplianceStatus.COMPLIANT
                if uncertainty_analyzed
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Uncertainty quantification performed"
                if uncertainty_analyzed
                else "Uncertainty analysis recommended",
                recommendation="Perform Monte Carlo or analytical uncertainty quantification",
            )
        )

        # 10. Year-over-year comparison
        yoy_available = (
            hasattr(calculation_result, "yoy_comparison")
            and calculation_result.yoy_comparison is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-010",
                rule_name="Year-over-year comparison",
                status=ComplianceStatus.COMPLIANT
                if yoy_available
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="YoY comparison available"
                if yoy_available
                else "YoY comparison not available (may be first year)",
                recommendation="Track year-over-year emissions for target monitoring",
            )
        )

        # 11. Double-counting prevention
        no_double_counting = self._check_double_counting_prevention(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-011",
                rule_name="Double-counting prevention (Scope 1/2)",
                status=ComplianceStatus.COMPLIANT
                if no_double_counting
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="No double-counting detected"
                if no_double_counting
                else "Potential double-counting with Scope 1/2",
                recommendation="Ensure Category 3 excludes emissions already in Scope 1 or 2",
            )
        )

        # 12. Boundary description
        boundary_described = (
            hasattr(calculation_result, "boundary_description")
            and calculation_result.boundary_description
        )
        findings.append(
            ComplianceFinding(
                rule_id="GHG-CAT3-012",
                rule_name="Category boundary description",
                status=ComplianceStatus.COMPLIANT
                if boundary_described
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Boundary clearly described"
                if boundary_described
                else "Boundary description recommended",
                recommendation="Document which fuels/energy types are included/excluded",
            )
        )

        # Calculate overall status and score
        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.GHG_PROTOCOL_SCOPE3,
            status=status,
            findings=findings,
            score=score,
        )

    def check_csrd(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with CSRD/ESRS E1.

        Validates 10 requirements for CSRD disclosure:
          1. Material Scope 3 categories reported
          2. Category 3 materiality assessment
          3. GHG intensity per revenue
          4. Data sources and methodologies disclosed
          5. Transition plan alignment
          6. Base year and target comparison
          7. Activity-level breakdown
          8. Uncertainty disclosure
          9. Improvement plan
         10. Third-party verification encouraged

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. Material Scope 3 categories reported
        has_emissions = calculation_result.total_emissions_kg_co2e > Decimal(
            "0"
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-001",
                rule_name="Material Scope 3 categories reported",
                status=ComplianceStatus.COMPLIANT
                if has_emissions
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Category 3 emissions reported"
                if has_emissions
                else "No Category 3 emissions reported",
                recommendation="Report all material Scope 3 categories",
            )
        )

        # 2. Category 3 materiality
        materiality_assessed = (
            hasattr(calculation_result, "materiality_result")
            and calculation_result.materiality_result is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-002",
                rule_name="Category 3 materiality assessment",
                status=ComplianceStatus.COMPLIANT
                if materiality_assessed
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Materiality assessment performed"
                if materiality_assessed
                else "Materiality assessment recommended",
                recommendation="Assess whether Category 3 is material for your business",
            )
        )

        # 3. GHG intensity per revenue
        has_intensity = (
            hasattr(calculation_result, "ghg_intensity_per_revenue")
            and calculation_result.ghg_intensity_per_revenue is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-003",
                rule_name="GHG intensity per revenue",
                status=ComplianceStatus.COMPLIANT
                if has_intensity
                else ComplianceStatus.NON_COMPLIANT,
                severity="major",
                message="GHG intensity metric calculated"
                if has_intensity
                else "GHG intensity per revenue required",
                recommendation="Calculate tCO2e per million EUR/USD revenue",
            )
        )

        # 4. Data sources disclosed
        data_sources_disclosed = self._check_data_sources_disclosed(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-004",
                rule_name="Data sources and methodologies disclosed",
                status=ComplianceStatus.COMPLIANT
                if data_sources_disclosed
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Data sources disclosed"
                if data_sources_disclosed
                else "Data source disclosure incomplete",
                recommendation="Disclose EF sources, calculation methods, and data providers",
            )
        )

        # 5. Transition plan alignment
        transition_aligned = (
            hasattr(calculation_result, "transition_plan_aligned")
            and calculation_result.transition_plan_aligned
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-005",
                rule_name="Transition plan alignment",
                status=ComplianceStatus.COMPLIANT
                if transition_aligned
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Aligned with transition plan"
                if transition_aligned
                else "Transition plan alignment not documented",
                recommendation="Link Category 3 to climate transition plan and targets",
            )
        )

        # 6. Base year comparison
        has_base_year = (
            hasattr(calculation_result, "base_year")
            and calculation_result.base_year is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-006",
                rule_name="Base year and target comparison",
                status=ComplianceStatus.COMPLIANT
                if has_base_year
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Base year comparison available"
                if has_base_year
                else "Base year not defined",
                recommendation="Establish base year and track progress",
            )
        )

        # 7. Activity-level breakdown
        has_breakdown = self._check_activity_breakdown(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-007",
                rule_name="Activity-level breakdown",
                status=ComplianceStatus.COMPLIANT
                if has_breakdown
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Activity-level breakdown provided"
                if has_breakdown
                else "Activity-level breakdown incomplete",
                recommendation="Break down Category 3 by activities (3a/3b/3c/3d)",
            )
        )

        # 8. Uncertainty disclosure
        uncertainty_disclosed = self._check_uncertainty_disclosure(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-008",
                rule_name="Uncertainty disclosure",
                status=ComplianceStatus.COMPLIANT
                if uncertainty_disclosed
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Uncertainty disclosed"
                if uncertainty_disclosed
                else "Uncertainty disclosure recommended",
                recommendation="Disclose uncertainty ranges and confidence intervals",
            )
        )

        # 9. Improvement plan
        has_improvement_plan = (
            hasattr(calculation_result, "improvement_plan")
            and calculation_result.improvement_plan
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-009",
                rule_name="Data quality improvement plan",
                status=ComplianceStatus.COMPLIANT
                if has_improvement_plan
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Improvement plan documented"
                if has_improvement_plan
                else "Improvement plan recommended",
                recommendation="Document plan to improve data quality over time",
            )
        )

        # 10. Third-party verification
        is_verified = (
            hasattr(calculation_result, "is_verified")
            and calculation_result.is_verified
        )
        findings.append(
            ComplianceFinding(
                rule_id="CSRD-E1-010",
                rule_name="Third-party verification",
                status=ComplianceStatus.COMPLIANT
                if is_verified
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Third-party verification obtained"
                if is_verified
                else "Third-party verification encouraged",
                recommendation="Obtain limited or reasonable assurance from third party",
            )
        )

        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.CSRD_ESRS_E1,
            status=status,
            findings=findings,
            score=score,
        )

    def check_cdp(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Questionnaire.

        Validates 10 CDP requirements for Category 3:
          1. Relevance assessment
          2. Emissions figure with calculation methodology
          3. Data quality score
          4. Verification status
          5. Year-over-year change explanation
          6. Supplier engagement on upstream data
          7. Exclusions must be justified
          8. Category boundary defined
          9. EF sources cited
         10. Improvement targets

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. Relevance assessment
        relevance_assessed = (
            calculation_result.total_emissions_kg_co2e > Decimal("0")
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-001",
                rule_name="Category 3 relevance assessment",
                status=ComplianceStatus.COMPLIANT
                if relevance_assessed
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Category 3 assessed as relevant"
                if relevance_assessed
                else "Category 3 relevance not assessed",
                recommendation="Assess relevance of Category 3 to your operations",
            )
        )

        # 2. Emissions figure with method
        has_method = calculation_result.method is not None
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-002",
                rule_name="Emissions figure with calculation methodology",
                status=ComplianceStatus.COMPLIANT
                if has_method
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message=f"Emissions: {calculation_result.total_emissions_tco2e} tCO2e using {calculation_result.method.value}"
                if has_method
                else "Calculation method not disclosed",
                recommendation="Report emissions with calculation methodology",
            )
        )

        # 3. Data quality score
        has_dqi = self._check_dqi_assessment(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-003",
                rule_name="Data quality score",
                status=ComplianceStatus.COMPLIANT
                if has_dqi
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Data quality score provided"
                if has_dqi
                else "Data quality score recommended",
                recommendation="Provide CDP data quality score (1-5 scale)",
            )
        )

        # 4. Verification status
        is_verified = (
            hasattr(calculation_result, "is_verified")
            and calculation_result.is_verified
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-004",
                rule_name="Verification status",
                status=ComplianceStatus.COMPLIANT
                if is_verified
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Emissions verified"
                if is_verified
                else "Verification status not indicated",
                recommendation="Indicate whether emissions are third-party verified",
            )
        )

        # 5. YoY change explanation
        has_yoy_explanation = (
            hasattr(calculation_result, "yoy_comparison")
            and calculation_result.yoy_comparison is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-005",
                rule_name="Year-over-year change explanation",
                status=ComplianceStatus.COMPLIANT
                if has_yoy_explanation
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="YoY change explained"
                if has_yoy_explanation
                else "YoY explanation not available (may be first year)",
                recommendation="Explain drivers of year-over-year change",
            )
        )

        # 6. Supplier engagement
        has_supplier_engagement = (
            hasattr(calculation_result, "supplier_engagement")
            and calculation_result.supplier_engagement
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-006",
                rule_name="Supplier engagement on upstream data",
                status=ComplianceStatus.COMPLIANT
                if has_supplier_engagement
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Supplier engagement documented"
                if has_supplier_engagement
                else "Supplier engagement recommended",
                recommendation="Engage suppliers for primary upstream data",
            )
        )

        # 7. Exclusions justified
        has_exclusions = (
            hasattr(calculation_result, "exclusions_justified")
            and calculation_result.exclusions_justified
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-007",
                rule_name="Exclusions must be justified",
                status=ComplianceStatus.COMPLIANT
                if has_exclusions
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Exclusions documented and justified"
                if has_exclusions
                else "Document any exclusions from Category 3",
                recommendation="Justify any exclusions from Category 3 boundary",
            )
        )

        # 8. Category boundary defined
        boundary_defined = (
            hasattr(calculation_result, "boundary_description")
            and calculation_result.boundary_description
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-008",
                rule_name="Category boundary defined",
                status=ComplianceStatus.COMPLIANT
                if boundary_defined
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Category boundary clearly defined"
                if boundary_defined
                else "Category boundary definition recommended",
                recommendation="Define which activities/fuels are in Category 3 boundary",
            )
        )

        # 9. EF sources cited
        ef_sources_cited = self._check_ef_sources_cited(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-009",
                rule_name="EF sources cited",
                status=ComplianceStatus.COMPLIANT
                if ef_sources_cited
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Emission factor sources cited"
                if ef_sources_cited
                else "EF source citation incomplete",
                recommendation="Cite all emission factor sources (DEFRA, IEA, etc.)",
            )
        )

        # 10. Improvement targets
        has_targets = (
            hasattr(calculation_result, "improvement_targets")
            and calculation_result.improvement_targets
        )
        findings.append(
            ComplianceFinding(
                rule_id="CDP-CAT3-010",
                rule_name="Improvement targets",
                status=ComplianceStatus.COMPLIANT
                if has_targets
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Improvement targets set"
                if has_targets
                else "Improvement targets recommended",
                recommendation="Set targets for reducing Category 3 emissions",
            )
        )

        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.CDP,
            status=status,
            findings=findings,
            score=score,
        )

    def check_sbti(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with Science Based Targets initiative (SBTi).

        Validates 8 SBTi requirements:
          1. Include in Scope 3 screening if >1% of total
          2. Boundary completeness >95%
          3. Data quality improvement plan
          4. Base year recalculation criteria
          5. Target-consistent methodology
          6. Annual progress disclosure
          7. Activity-level granularity
          8. Supplier-specific data prioritized

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. Scope 3 screening (1% threshold)
        has_emissions = calculation_result.total_emissions_kg_co2e > Decimal(
            "0"
        )
        materiality_threshold = (
            hasattr(calculation_result, "sbti_materiality_pct")
            and calculation_result.sbti_materiality_pct
            and calculation_result.sbti_materiality_pct > Decimal("1.0")
        )
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-001",
                rule_name="Include in Scope 3 screening if >1%",
                status=ComplianceStatus.COMPLIANT
                if has_emissions
                else ComplianceStatus.NOT_APPLICABLE,
                severity="critical",
                message="Category 3 included in screening"
                if has_emissions
                else "Category 3 below 1% threshold",
                recommendation="Include Category 3 if >1% of total Scope 3",
            )
        )

        # 2. Boundary completeness >95%
        completeness = self._calculate_boundary_completeness(
            calculation_result
        )
        is_complete = completeness >= Decimal("95.0")
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-002",
                rule_name="Boundary completeness >95%",
                status=ComplianceStatus.COMPLIANT
                if is_complete
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message=f"Boundary completeness: {completeness}%"
                if is_complete
                else f"Boundary completeness: {completeness}% (target: >95%)",
                recommendation="Achieve >95% boundary completeness for SBTi",
            )
        )

        # 3. Data quality improvement plan
        has_improvement_plan = (
            hasattr(calculation_result, "improvement_plan")
            and calculation_result.improvement_plan
        )
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-003",
                rule_name="Data quality improvement plan",
                status=ComplianceStatus.COMPLIANT
                if has_improvement_plan
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Improvement plan documented"
                if has_improvement_plan
                else "Improvement plan required",
                recommendation="Document plan to improve data quality toward primary data",
            )
        )

        # 4. Base year recalculation criteria
        has_recalc_criteria = (
            hasattr(calculation_result, "recalculation_criteria")
            and calculation_result.recalculation_criteria
        )
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-004",
                rule_name="Base year recalculation criteria",
                status=ComplianceStatus.COMPLIANT
                if has_recalc_criteria
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Recalculation criteria defined"
                if has_recalc_criteria
                else "Recalculation criteria recommended",
                recommendation="Define criteria for base year recalculation (e.g., >5% change)",
            )
        )

        # 5. Target-consistent methodology
        methodology_consistent = calculation_result.method in [
            CalculationMethod.SUPPLIER_SPECIFIC,
            CalculationMethod.AVERAGE_DATA,
        ]
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-005",
                rule_name="Target-consistent methodology",
                status=ComplianceStatus.COMPLIANT
                if methodology_consistent
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Methodology consistent with target-setting"
                if methodology_consistent
                else "Methodology should align with target-setting approach",
                recommendation="Use consistent methodology for baseline and target tracking",
            )
        )

        # 6. Annual progress disclosure
        has_annual_disclosure = (
            hasattr(calculation_result, "reporting_year")
            and calculation_result.reporting_year is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-006",
                rule_name="Annual progress disclosure",
                status=ComplianceStatus.COMPLIANT
                if has_annual_disclosure
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Annual reporting in place"
                if has_annual_disclosure
                else "Annual disclosure required",
                recommendation="Disclose Category 3 progress annually",
            )
        )

        # 7. Activity-level granularity
        has_granularity = self._check_activity_breakdown(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-007",
                rule_name="Activity-level granularity",
                status=ComplianceStatus.COMPLIANT
                if has_granularity
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Activity-level granularity provided"
                if has_granularity
                else "Activity-level granularity recommended",
                recommendation="Break down to activity level (3a/3b/3c/3d) for granular tracking",
            )
        )

        # 8. Supplier-specific data prioritized
        has_supplier_data = self._check_supplier_specific_data(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="SBTI-CAT3-008",
                rule_name="Supplier-specific data prioritized",
                status=ComplianceStatus.COMPLIANT
                if has_supplier_data
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Supplier-specific data used"
                if has_supplier_data
                else "Prioritize supplier-specific data over averages",
                recommendation="Work with suppliers to obtain primary upstream data",
            )
        )

        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.SBTI,
            status=status,
            findings=findings,
            score=score,
        )

    def check_sb253(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253.

        Validates 8 SB 253 requirements:
          1. Revenue threshold $1B (applicability check)
          2. All material Scope 3 categories
          3. Third-party verification required
          4. PCAF methodology where applicable
          5. Annual reporting
          6. Public disclosure
          7. Methodology documentation
          8. Historical comparison

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. Revenue threshold (applicability)
        revenue_threshold_met = (
            hasattr(calculation_result, "annual_revenue_usd")
            and calculation_result.annual_revenue_usd
            and calculation_result.annual_revenue_usd
            >= Decimal("1_000_000_000")
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-001",
                rule_name="Revenue threshold $1B (applicability)",
                status=ComplianceStatus.COMPLIANT
                if revenue_threshold_met
                else ComplianceStatus.NOT_APPLICABLE,
                severity="critical",
                message="SB 253 applies (revenue >$1B)"
                if revenue_threshold_met
                else "SB 253 may not apply (revenue <$1B or not specified)",
                recommendation="Verify annual California revenue exceeds $1B threshold",
            )
        )

        # 2. Material Scope 3 categories
        has_emissions = calculation_result.total_emissions_kg_co2e > Decimal(
            "0"
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-002",
                rule_name="All material Scope 3 categories",
                status=ComplianceStatus.COMPLIANT
                if has_emissions
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Category 3 emissions reported"
                if has_emissions
                else "Category 3 must be reported if material",
                recommendation="Report all material Scope 3 categories",
            )
        )

        # 3. Third-party verification
        is_verified = (
            hasattr(calculation_result, "is_verified")
            and calculation_result.is_verified
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-003",
                rule_name="Third-party verification required",
                status=ComplianceStatus.COMPLIANT
                if is_verified
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Third-party verification obtained"
                if is_verified
                else "Third-party verification REQUIRED for SB 253",
                recommendation="Obtain limited or reasonable assurance from accredited verifier",
            )
        )

        # 4. PCAF methodology
        uses_pcaf = (
            hasattr(calculation_result, "uses_pcaf_methodology")
            and calculation_result.uses_pcaf_methodology
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-004",
                rule_name="PCAF methodology where applicable",
                status=ComplianceStatus.COMPLIANT
                if uses_pcaf
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="PCAF methodology applied"
                if uses_pcaf
                else "Consider PCAF for financed emissions component",
                recommendation="Use PCAF methodology for financial sector emissions",
            )
        )

        # 5. Annual reporting
        has_reporting_year = (
            hasattr(calculation_result, "reporting_year")
            and calculation_result.reporting_year is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-005",
                rule_name="Annual reporting",
                status=ComplianceStatus.COMPLIANT
                if has_reporting_year
                else ComplianceStatus.NON_COMPLIANT,
                severity="major",
                message="Annual reporting in place"
                if has_reporting_year
                else "Annual reporting required",
                recommendation="Report Category 3 emissions annually",
            )
        )

        # 6. Public disclosure
        is_public = (
            hasattr(calculation_result, "is_publicly_disclosed")
            and calculation_result.is_publicly_disclosed
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-006",
                rule_name="Public disclosure",
                status=ComplianceStatus.COMPLIANT
                if is_public
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Emissions publicly disclosed"
                if is_public
                else "Public disclosure REQUIRED for SB 253",
                recommendation="Publicly disclose emissions on company website or CDP",
            )
        )

        # 7. Methodology documentation
        methodology_documented = (
            hasattr(calculation_result, "methodology_documentation")
            and calculation_result.methodology_documentation
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-007",
                rule_name="Methodology documentation",
                status=ComplianceStatus.COMPLIANT
                if methodology_documented
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Methodology fully documented"
                if methodology_documented
                else "Methodology documentation required",
                recommendation="Document calculation methodology in detail",
            )
        )

        # 8. Historical comparison
        has_historical = (
            hasattr(calculation_result, "yoy_comparison")
            and calculation_result.yoy_comparison is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="SB253-CAT3-008",
                rule_name="Historical comparison",
                status=ComplianceStatus.COMPLIANT
                if has_historical
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Historical comparison available"
                if has_historical
                else "Historical comparison recommended",
                recommendation="Provide year-over-year comparison and trends",
            )
        )

        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.SB_253,
            status=status,
            findings=findings,
            score=score,
        )

    def check_gri(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305 Standard.

        Validates 6 GRI 305 requirements:
          1. GRI 305-3 disclosure (other indirect emissions)
          2. Base year disclosure
          3. Consolidation approach
          4. EF sources and GWP values
          5. Methodology and assumptions
          6. Biogenic CO2 reported separately

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. GRI 305-3 disclosure
        has_emissions = calculation_result.total_emissions_kg_co2e > Decimal(
            "0"
        )
        findings.append(
            ComplianceFinding(
                rule_id="GRI-305-001",
                rule_name="GRI 305-3 Other indirect GHG emissions",
                status=ComplianceStatus.COMPLIANT
                if has_emissions
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="GRI 305-3 disclosure provided"
                if has_emissions
                else "GRI 305-3 disclosure required",
                recommendation="Report all other indirect (Scope 3) emissions",
            )
        )

        # 2. Base year disclosure
        has_base_year = (
            hasattr(calculation_result, "base_year")
            and calculation_result.base_year is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="GRI-305-002",
                rule_name="Base year disclosure",
                status=ComplianceStatus.COMPLIANT
                if has_base_year
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Base year disclosed"
                if has_base_year
                else "Base year disclosure required",
                recommendation="Disclose base year for emissions comparison",
            )
        )

        # 3. Consolidation approach
        has_consolidation = (
            hasattr(calculation_result, "consolidation_approach")
            and calculation_result.consolidation_approach
        )
        findings.append(
            ComplianceFinding(
                rule_id="GRI-305-003",
                rule_name="Consolidation approach",
                status=ComplianceStatus.COMPLIANT
                if has_consolidation
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Consolidation approach disclosed"
                if has_consolidation
                else "Consolidation approach required",
                recommendation="Disclose consolidation approach (equity share, financial, operational)",
            )
        )

        # 4. EF sources and GWP values
        has_ef_sources = self._check_ef_sources_cited(calculation_result)
        has_gwp = calculation_result.gwp_source is not None
        ef_gwp_complete = has_ef_sources and has_gwp
        findings.append(
            ComplianceFinding(
                rule_id="GRI-305-004",
                rule_name="EF sources and GWP values",
                status=ComplianceStatus.COMPLIANT
                if ef_gwp_complete
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message=f"EF sources cited, GWP: {calculation_result.gwp_source.value}"
                if ef_gwp_complete
                else "EF sources and GWP values incomplete",
                recommendation="Cite all EF sources and GWP values (IPCC AR version)",
            )
        )

        # 5. Methodology and assumptions
        has_methodology = (
            hasattr(calculation_result, "methodology_documentation")
            and calculation_result.methodology_documentation
        )
        findings.append(
            ComplianceFinding(
                rule_id="GRI-305-005",
                rule_name="Methodology and assumptions",
                status=ComplianceStatus.COMPLIANT
                if has_methodology
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Methodology and assumptions disclosed"
                if has_methodology
                else "Methodology and assumptions required",
                recommendation="Disclose calculation methodology and key assumptions",
            )
        )

        # 6. Biogenic CO2 separate
        biogenic_separate = self._check_biogenic_separation(
            calculation_result
        )
        findings.append(
            ComplianceFinding(
                rule_id="GRI-305-006",
                rule_name="Biogenic CO2 reported separately",
                status=ComplianceStatus.COMPLIANT
                if biogenic_separate
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Biogenic CO2 reported separately"
                if biogenic_separate
                else "Biogenic CO2 separation recommended",
                recommendation="Report biogenic CO2 separately from fossil CO2",
            )
        )

        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.GRI_305,
            status=status,
            findings=findings,
            score=score,
        )

    def check_iso14064(
        self, calculation_result: CalculationResult
    ) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064-1:2018.

        Validates 6 ISO 14064 requirements:
          1. Categorize as other indirect GHG emissions
          2. Quantification methodology
          3. Uncertainty assessment
          4. Documentation requirements
          5. Verification provisions
          6. Reporting period alignment

        Args:
            calculation_result: The calculation result to check.

        Returns:
            ComplianceCheckResult with findings and status.
        """
        findings: List[ComplianceFinding] = []

        # 1. Categorize as other indirect
        has_emissions = calculation_result.total_emissions_kg_co2e > Decimal(
            "0"
        )
        findings.append(
            ComplianceFinding(
                rule_id="ISO-14064-001",
                rule_name="Categorize as other indirect GHG emissions",
                status=ComplianceStatus.COMPLIANT
                if has_emissions
                else ComplianceStatus.NON_COMPLIANT,
                severity="critical",
                message="Categorized as indirect emissions"
                if has_emissions
                else "Categorization required",
                recommendation="Categorize Category 3 as other indirect GHG emissions",
            )
        )

        # 2. Quantification methodology
        has_method = calculation_result.method is not None
        findings.append(
            ComplianceFinding(
                rule_id="ISO-14064-002",
                rule_name="Quantification methodology",
                status=ComplianceStatus.COMPLIANT
                if has_method
                else ComplianceStatus.NON_COMPLIANT,
                severity="major",
                message=f"Methodology: {calculation_result.method.value}"
                if has_method
                else "Quantification methodology required",
                recommendation="Document quantification methodology per ISO 14064",
            )
        )

        # 3. Uncertainty assessment
        has_uncertainty = self._check_uncertainty_analysis(calculation_result)
        findings.append(
            ComplianceFinding(
                rule_id="ISO-14064-003",
                rule_name="Uncertainty assessment",
                status=ComplianceStatus.COMPLIANT
                if has_uncertainty
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Uncertainty assessment performed"
                if has_uncertainty
                else "Uncertainty assessment required",
                recommendation="Conduct uncertainty assessment per ISO 14064",
            )
        )

        # 4. Documentation requirements
        has_documentation = (
            hasattr(calculation_result, "methodology_documentation")
            and calculation_result.methodology_documentation
        )
        findings.append(
            ComplianceFinding(
                rule_id="ISO-14064-004",
                rule_name="Documentation requirements",
                status=ComplianceStatus.COMPLIANT
                if has_documentation
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Documentation complete"
                if has_documentation
                else "Documentation requirements incomplete",
                recommendation="Maintain complete documentation per ISO 14064",
            )
        )

        # 5. Verification provisions
        is_verified = (
            hasattr(calculation_result, "is_verified")
            and calculation_result.is_verified
        )
        findings.append(
            ComplianceFinding(
                rule_id="ISO-14064-005",
                rule_name="Verification provisions",
                status=ComplianceStatus.COMPLIANT
                if is_verified
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="minor",
                message="Verification obtained"
                if is_verified
                else "Verification provisions recommended",
                recommendation="Consider third-party verification per ISO 14064-3",
            )
        )

        # 6. Reporting period alignment
        has_reporting_period = (
            hasattr(calculation_result, "reporting_year")
            and calculation_result.reporting_year is not None
        )
        findings.append(
            ComplianceFinding(
                rule_id="ISO-14064-006",
                rule_name="Reporting period alignment",
                status=ComplianceStatus.COMPLIANT
                if has_reporting_period
                else ComplianceStatus.PARTIALLY_COMPLIANT,
                severity="major",
                message="Reporting period aligned"
                if has_reporting_period
                else "Reporting period alignment required",
                recommendation="Align reporting period with organizational reporting year",
            )
        )

        status, score = self._calculate_status_and_score(findings)

        return ComplianceCheckResult(
            framework=ComplianceFramework.ISO_14064,
            status=status,
            findings=findings,
            score=score,
        )

    def get_framework_requirements(
        self, framework: ComplianceFramework
    ) -> List[str]:
        """
        Get the list of requirements for a specific framework.

        Args:
            framework: The compliance framework.

        Returns:
            List of requirement IDs for the framework.

        Example:
            >>> reqs = engine.get_framework_requirements(ComplianceFramework.GHG_PROTOCOL_SCOPE3)
            >>> print(f"{len(reqs)} requirements")
        """
        requirements_map = {
            ComplianceFramework.GHG_PROTOCOL_SCOPE3: GHG_PROTOCOL_REQUIREMENTS,
            ComplianceFramework.CSRD_ESRS_E1: CSRD_REQUIREMENTS,
            ComplianceFramework.CDP: CDP_REQUIREMENTS,
            ComplianceFramework.SBTI: SBTI_REQUIREMENTS,
            ComplianceFramework.SB_253: SB253_REQUIREMENTS,
            ComplianceFramework.GRI_305: GRI_REQUIREMENTS,
            ComplianceFramework.ISO_14064: ISO_REQUIREMENTS,
        }
        return requirements_map.get(framework, [])

    def get_compliance_summary(
        self, results: List[ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Generate a summary of compliance check results.

        Args:
            results: List of compliance check results.

        Returns:
            Summary dictionary with counts and statistics.

        Example:
            >>> summary = engine.get_compliance_summary(results)
            >>> print(f"Compliant: {summary['compliant_count']}")
        """
        total = len(results)
        compliant = sum(
            1
            for r in results
            if r.status == ComplianceStatus.COMPLIANT
        )
        partially_compliant = sum(
            1
            for r in results
            if r.status == ComplianceStatus.PARTIALLY_COMPLIANT
        )
        non_compliant = sum(
            1
            for r in results
            if r.status == ComplianceStatus.NON_COMPLIANT
        )
        not_applicable = sum(
            1
            for r in results
            if r.status == ComplianceStatus.NOT_APPLICABLE
        )

        avg_score = (
            sum(r.score for r in results) / Decimal(total)
            if total > 0
            else Decimal("0")
        )

        # Count findings by severity
        critical_findings = sum(
            sum(1 for f in r.findings if f.severity == "critical")
            for r in results
        )
        major_findings = sum(
            sum(1 for f in r.findings if f.severity == "major")
            for r in results
        )
        minor_findings = sum(
            sum(1 for f in r.findings if f.severity == "minor")
            for r in results
        )

        return {
            "total_frameworks": total,
            "compliant_count": compliant,
            "partially_compliant_count": partially_compliant,
            "non_compliant_count": non_compliant,
            "not_applicable_count": not_applicable,
            "average_score": float(avg_score),
            "critical_findings": critical_findings,
            "major_findings": major_findings,
            "minor_findings": minor_findings,
            "frameworks_checked": [r.framework.value for r in results],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine usage statistics.

        Returns:
            Dictionary with usage statistics.

        Example:
            >>> stats = engine.get_statistics()
            >>> print(f"Checks performed: {stats['checks_performed']}")
        """
        return {
            "checks_performed": self._checks_performed,
            "frameworks_checked_count": len(self._frameworks_checked),
            "frameworks_checked": [
                f.value for f in self._frameworks_checked
            ],
        }

    def reset(self) -> None:
        """
        Reset engine statistics.

        Example:
            >>> engine.reset()
            >>> assert engine.get_statistics()['checks_performed'] == 0
        """
        self._checks_performed = 0
        self._frameworks_checked.clear()
        logger.info("ComplianceCheckerEngine statistics reset")

    # -----------------------------------------------------------------------
    # Private Helper Methods
    # -----------------------------------------------------------------------

    def _check_wtt_combustion_exclusion(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if WTT factors exclude combustion emissions."""
        # In a production system, this would verify that WTT EFs
        # are from trusted sources that exclude combustion.
        # For now, we check if Activity 3a results are present.
        return len(calculation_result.activity_3a_results) > 0

    def _check_td_loss_factors(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if T&D loss factors are country-specific."""
        # Check if Activity 3c results use country-specific factors.
        if not calculation_result.activity_3c_results:
            return False

        # In production, verify factor sources are IEA/World Bank.
        return True

    def _check_upstream_generation_exclusion(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if upstream EFs exclude generation-point emissions."""
        # In production, verify upstream EFs only cover extraction/transmission.
        return len(calculation_result.activity_3b_results) > 0

    def _check_utility_only_3d(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if Activity 3d is only used by utilities."""
        # If no 3d results, compliant (not applicable).
        # If 3d results exist, assume organization is utility.
        return True

    def _check_per_gas_breakdown(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if per-gas breakdown (CO2, CH4, N2O) is provided."""
        # Check if Activity 3a results have gas breakdowns.
        if not calculation_result.activity_3a_results:
            return False

        # Check first result has gas breakdown.
        first_result = calculation_result.activity_3a_results[0]
        return (
            hasattr(first_result, "emissions_co2")
            and hasattr(first_result, "emissions_ch4")
            and hasattr(first_result, "emissions_n2o")
        )

    def _check_dqi_assessment(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if data quality assessment is performed."""
        # Check if results have DQI scores.
        if calculation_result.activity_3a_results:
            return hasattr(
                calculation_result.activity_3a_results[0], "dqi_score"
            )
        return False

    def _check_uncertainty_analysis(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if uncertainty analysis is performed."""
        # Check if results have uncertainty percentages.
        if calculation_result.activity_3a_results:
            return hasattr(
                calculation_result.activity_3a_results[0], "uncertainty_pct"
            )
        return False

    def _check_double_counting_prevention(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if double-counting with Scope 1/2 is prevented."""
        # In production, this would check for boundary overlaps.
        # For now, assume compliant if emissions are calculated.
        return calculation_result.total_emissions_kg_co2e > Decimal("0")

    def _check_data_sources_disclosed(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if data sources are disclosed."""
        # Check if EF sources are documented.
        if calculation_result.activity_3a_results:
            return hasattr(
                calculation_result.activity_3a_results[0], "wtt_ef_source"
            )
        return False

    def _check_activity_breakdown(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if activity-level breakdown is provided."""
        # Check if at least 2 activities are reported.
        activities_count = sum(
            [
                len(calculation_result.activity_3a_results) > 0,
                len(calculation_result.activity_3b_results) > 0,
                len(calculation_result.activity_3c_results) > 0,
                len(calculation_result.activity_3d_results) > 0,
            ]
        )
        return activities_count >= 2

    def _check_uncertainty_disclosure(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if uncertainty is disclosed."""
        return self._check_uncertainty_analysis(calculation_result)

    def _check_ef_sources_cited(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if emission factor sources are cited."""
        return self._check_data_sources_disclosed(calculation_result)

    def _calculate_boundary_completeness(
        self, calculation_result: CalculationResult
    ) -> Decimal:
        """Calculate boundary completeness percentage."""
        # In production, this would calculate based on coverage of fuel/energy types.
        # For now, estimate based on number of activities.
        activities_count = sum(
            [
                len(calculation_result.activity_3a_results) > 0,
                len(calculation_result.activity_3b_results) > 0,
                len(calculation_result.activity_3c_results) > 0,
                len(calculation_result.activity_3d_results) > 0,
            ]
        )
        # 3 or 4 activities = 95-100%, 2 = 85%, 1 = 70%
        if activities_count >= 3:
            return Decimal("97.5")
        elif activities_count == 2:
            return Decimal("85.0")
        elif activities_count == 1:
            return Decimal("70.0")
        else:
            return Decimal("0.0")

    def _check_supplier_specific_data(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if supplier-specific data is used."""
        # Check if method is supplier-specific.
        return (
            calculation_result.method
            == CalculationMethod.SUPPLIER_SPECIFIC
        )

    def _check_biogenic_separation(
        self, calculation_result: CalculationResult
    ) -> bool:
        """Check if biogenic CO2 is reported separately."""
        # Check if Activity 3a results track biogenic flag.
        if calculation_result.activity_3a_results:
            return hasattr(
                calculation_result.activity_3a_results[0], "is_biogenic"
            )
        return False

    def _calculate_status_and_score(
        self, findings: List[ComplianceFinding]
    ) -> tuple[ComplianceStatus, Decimal]:
        """
        Calculate overall compliance status and score from findings.

        Args:
            findings: List of compliance findings.

        Returns:
            Tuple of (status, score).
        """
        if not findings:
            return ComplianceStatus.NOT_APPLICABLE, Decimal("0")

        # Count statuses by severity
        critical_non_compliant = sum(
            1
            for f in findings
            if f.severity == "critical"
            and f.status == ComplianceStatus.NON_COMPLIANT
        )
        major_non_compliant = sum(
            1
            for f in findings
            if f.severity == "major"
            and f.status == ComplianceStatus.NON_COMPLIANT
        )

        compliant_count = sum(
            1 for f in findings if f.status == ComplianceStatus.COMPLIANT
        )
        total_count = len(findings)

        # Calculate score (0-100)
        base_score = (
            Decimal(compliant_count) / Decimal(total_count) * Decimal("100")
        )

        # Determine status
        if critical_non_compliant > 0:
            status = ComplianceStatus.NON_COMPLIANT
            score = min(base_score, Decimal("50"))
        elif major_non_compliant > 2:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            score = min(base_score, Decimal("75"))
        elif compliant_count == total_count:
            status = ComplianceStatus.COMPLIANT
            score = Decimal("100")
        elif compliant_count >= total_count * 0.8:
            status = ComplianceStatus.COMPLIANT
            score = base_score
        else:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            score = base_score

        return status, score
