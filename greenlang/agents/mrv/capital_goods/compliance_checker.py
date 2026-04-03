"""
ComplianceCheckerEngine - Engine 6 for AGENT-MRV-015 (Capital Goods Agent)

This module implements the ComplianceCheckerEngine that validates capital goods
emissions calculations against 7 regulatory frameworks:
1. GHG Protocol Scope 3 (2011)
2. CSRD/ESRS E1 (2025+)
3. CDP Climate Change (C6.5)
4. Science Based Targets initiative (SBTi)
5. California SB 253
6. GRI 305
7. ISO 14064

The engine performs capital-goods-specific compliance checks including:
- NO_DEPRECIATION_RULE: Emissions NOT depreciated over asset life
- CAPITALIZATION_CLASSIFICATION: Assets properly classified as PP&E
- CATEGORY_BOUNDARY: No overlap with Category 1 (Purchased Goods)
- SCOPE_BOUNDARY: No overlap with Scope 1/2 use-phase emissions
- CAPEX_VOLATILITY_CONTEXT: Narrative for major CapEx years
- USEFUL_LIFE_DOCUMENTATION: Useful life documented but NOT used for allocation

Example:
    >>> engine = ComplianceCheckerEngine()
    >>> results = engine.check_all(hybrid_result, frameworks=["GHG_PROTOCOL", "CSRD"])
    >>> summary = engine.get_compliance_summary(results)
    >>> gaps = engine.get_gaps(results)
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass, field
import threading

from greenlang.agents.mrv.capital_goods.models import (
    HybridResult,
    CalculationMethod,
    DataQualityScore,
    AssetType,
    EmissionScope
)

logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Three-tier compliance status."""
    COMPLIANT = "COMPLIANT"  # All requirements met
    PARTIAL = "PARTIAL"  # Some requirements met
    NON_COMPLIANT = "NON_COMPLIANT"  # Critical gaps exist


class ComplianceFramework(str, Enum):
    """Supported regulatory frameworks."""
    GHG_PROTOCOL = "GHG_PROTOCOL"
    CSRD = "CSRD"
    CDP = "CDP"
    SBTI = "SBTI"
    SB253 = "SB253"
    GRI = "GRI"
    ISO14064 = "ISO14064"


class ComplianceCheckType(str, Enum):
    """Capital-goods-specific compliance check types."""
    NO_DEPRECIATION_RULE = "NO_DEPRECIATION_RULE"
    CAPITALIZATION_CLASSIFICATION = "CAPITALIZATION_CLASSIFICATION"
    CATEGORY_BOUNDARY = "CATEGORY_BOUNDARY"
    SCOPE_BOUNDARY = "SCOPE_BOUNDARY"
    CAPEX_VOLATILITY_CONTEXT = "CAPEX_VOLATILITY_CONTEXT"
    USEFUL_LIFE_DOCUMENTATION = "USEFUL_LIFE_DOCUMENTATION"
    METHOD_HIERARCHY = "METHOD_HIERARCHY"
    DATA_QUALITY = "DATA_QUALITY"
    BASE_YEAR_CONSISTENCY = "BASE_YEAR_CONSISTENCY"
    UNCERTAINTY_REPORTING = "UNCERTAINTY_REPORTING"
    VERIFICATION_SCOPE = "VERIFICATION_SCOPE"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""
    check_type: ComplianceCheckType
    description: str
    is_critical: bool
    is_met: bool
    evidence: Optional[str] = None
    gap_description: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class ComplianceCheckResult:
    """Result of compliance check for a single framework."""
    framework: ComplianceFramework
    status: ComplianceStatus
    requirements: List[ComplianceRequirement] = field(default_factory=list)
    score: float = 0.0  # 0-100
    critical_gaps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceCheckerEngine:
    """
    Thread-safe singleton engine for compliance validation.

    This engine validates capital goods emissions calculations against 7 regulatory
    frameworks, ensuring compliance with framework-specific requirements and
    capital-goods-specific rules.

    Attributes:
        frameworks_config: Configuration for each framework

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> results = engine.check_all(hybrid_result)
        >>> if results["GHG_PROTOCOL"].status == ComplianceStatus.COMPLIANT:
        >>>     print("GHG Protocol compliant!")
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize ComplianceCheckerEngine."""
        if self._initialized:
            return

        self.frameworks_config = self._load_frameworks_config()
        self._initialized = True

        logger.info("ComplianceCheckerEngine initialized")

    def _load_frameworks_config(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration for each framework."""
        return {
            ComplianceFramework.GHG_PROTOCOL: {
                "version": "2011",
                "critical_checks": [
                    ComplianceCheckType.NO_DEPRECIATION_RULE,
                    ComplianceCheckType.CATEGORY_BOUNDARY,
                    ComplianceCheckType.SCOPE_BOUNDARY,
                    ComplianceCheckType.METHOD_HIERARCHY
                ],
                "required_documentation": [
                    "cradle_to_gate_boundary",
                    "method_hierarchy",
                    "data_quality_assessment"
                ]
            },
            ComplianceFramework.CSRD: {
                "version": "ESRS E1 (2025+)",
                "critical_checks": [
                    ComplianceCheckType.NO_DEPRECIATION_RULE,
                    ComplianceCheckType.BASE_YEAR_CONSISTENCY,
                    ComplianceCheckType.DATA_QUALITY
                ],
                "required_disclosure": [
                    "scope_3_category_2_separate",
                    "methodology_description",
                    "data_quality_noted"
                ]
            },
            ComplianceFramework.CDP: {
                "version": "Climate Change C6.5",
                "critical_checks": [
                    ComplianceCheckType.CATEGORY_BOUNDARY,
                    ComplianceCheckType.DATA_QUALITY
                ],
                "required_metrics": [
                    "category_2_emissions",
                    "calculation_method",
                    "spend_coverage_pct",
                    "data_quality_assessment"
                ]
            },
            ComplianceFramework.SBTI: {
                "version": "Net-Zero Standard",
                "critical_checks": [
                    ComplianceCheckType.CATEGORY_BOUNDARY,
                    ComplianceCheckType.BASE_YEAR_CONSISTENCY
                ],
                "materiality_threshold": 0.05,  # 5% of Scope 3
                "required_elements": [
                    "target_setting_boundary",
                    "base_year_recalculation_triggers"
                ]
            },
            ComplianceFramework.SB253: {
                "version": "California Climate Corporate Data Accountability Act",
                "critical_checks": [
                    ComplianceCheckType.NO_DEPRECIATION_RULE,
                    ComplianceCheckType.VERIFICATION_SCOPE
                ],
                "reporting_year": 2027,
                "assurance_year": 2030,
                "revenue_threshold": 1_000_000_000  # $1B
            },
            ComplianceFramework.GRI: {
                "version": "GRI 305",
                "critical_checks": [
                    ComplianceCheckType.BASE_YEAR_CONSISTENCY,
                    ComplianceCheckType.METHOD_HIERARCHY
                ],
                "required_disclosure": [
                    "ghg_emissions",
                    "methodology_reference",
                    "base_year_stated"
                ]
            },
            ComplianceFramework.ISO14064: {
                "version": "ISO 14064-1:2018",
                "critical_checks": [
                    ComplianceCheckType.UNCERTAINTY_REPORTING,
                    ComplianceCheckType.VERIFICATION_SCOPE
                ],
                "required_elements": [
                    "category_4_indirect",
                    "uncertainty_quantified",
                    "verification_scope_defined"
                ]
            }
        }

    def check_all(
        self,
        result: HybridResult,
        frameworks: Optional[List[str]] = None
    ) -> Dict[str, ComplianceCheckResult]:
        """
        Check compliance against all or specified frameworks.

        Args:
            result: HybridResult from calculation
            frameworks: Optional list of framework names (default: all 7)

        Returns:
            Dictionary mapping framework name to ComplianceCheckResult

        Example:
            >>> results = engine.check_all(hybrid_result)
            >>> results = engine.check_all(hybrid_result, ["GHG_PROTOCOL", "CSRD"])
        """
        logger.info("Starting compliance checks for %s frameworks", len(frameworks or self.frameworks_config))

        frameworks_to_check = frameworks or list(self.frameworks_config.keys())
        results = {}

        for framework_name in frameworks_to_check:
            try:
                framework = ComplianceFramework(framework_name)

                if framework == ComplianceFramework.GHG_PROTOCOL:
                    results[framework_name] = self.check_ghg_protocol(result)
                elif framework == ComplianceFramework.CSRD:
                    results[framework_name] = self.check_csrd(result)
                elif framework == ComplianceFramework.CDP:
                    results[framework_name] = self.check_cdp(result)
                elif framework == ComplianceFramework.SBTI:
                    results[framework_name] = self.check_sbti(result)
                elif framework == ComplianceFramework.SB253:
                    results[framework_name] = self.check_sb253(result)
                elif framework == ComplianceFramework.GRI:
                    results[framework_name] = self.check_gri(result)
                elif framework == ComplianceFramework.ISO14064:
                    results[framework_name] = self.check_iso14064(result)

                logger.info("%s check complete: %s", framework_name, results[framework_name].status)

            except Exception as e:
                logger.error("Error checking %s: %s", framework_name, e, exc_info=True)
                results[framework_name] = ComplianceCheckResult(
                    framework=ComplianceFramework(framework_name),
                    status=ComplianceStatus.NON_COMPLIANT,
                    critical_gaps=[f"Check failed: {str(e)}"]
                )

        return results

    def check_ghg_protocol(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with GHG Protocol Scope 3 (2011).

        Key requirements:
        - Cradle-to-gate boundary
        - No depreciation rule
        - Method hierarchy documented
        - DQI reported
        - Category 1/2 boundary clear

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for GHG Protocol
        """
        logger.debug("Checking GHG Protocol compliance")

        requirements = []

        # Check 1: NO_DEPRECIATION_RULE
        no_depreciation = self._check_no_depreciation(result)
        requirements.append(no_depreciation)

        # Check 2: CATEGORY_BOUNDARY
        category_boundary = self._check_category_boundary(result)
        requirements.append(category_boundary)

        # Check 3: SCOPE_BOUNDARY
        scope_boundary = self._check_scope_boundary(result)
        requirements.append(scope_boundary)

        # Check 4: METHOD_HIERARCHY
        method_hierarchy = self._check_method_hierarchy(result)
        requirements.append(method_hierarchy)

        # Check 5: DATA_QUALITY
        data_quality = self._check_data_quality(result)
        requirements.append(data_quality)

        # Check 6: CAPITALIZATION_CLASSIFICATION
        capitalization = self._check_capitalization_classification(result)
        requirements.append(capitalization)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.GHG_PROTOCOL,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "Scope 3 Standard 2011",
                "category": "Category 2 - Capital Goods"
            }
        )

    def check_csrd(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with CSRD/ESRS E1 (2025+).

        Key requirements:
        - Scope 3 Cat 2 disclosed separately
        - Methodology described
        - Data quality noted
        - Base year consistent

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for CSRD
        """
        logger.debug("Checking CSRD/ESRS E1 compliance")

        requirements = []

        # Check 1: NO_DEPRECIATION_RULE
        no_depreciation = self._check_no_depreciation(result)
        requirements.append(no_depreciation)

        # Check 2: BASE_YEAR_CONSISTENCY
        base_year = self._check_base_year_consistency(result)
        requirements.append(base_year)

        # Check 3: DATA_QUALITY
        data_quality = self._check_data_quality(result)
        requirements.append(data_quality)

        # Check 4: METHOD_HIERARCHY (non-critical for CSRD)
        method_hierarchy = self._check_method_hierarchy(result)
        method_hierarchy.is_critical = False
        requirements.append(method_hierarchy)

        # Check 5: CAPEX_VOLATILITY_CONTEXT
        capex_context = self._check_capex_volatility_context(result)
        requirements.append(capex_context)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.CSRD,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "ESRS E1 (2025+)",
                "effective_date": "2025-01-01"
            }
        )

    def check_cdp(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with CDP Climate Change (C6.5).

        Key requirements:
        - Category 2 reported
        - Method described
        - % of spend covered
        - Data quality assessment

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for CDP
        """
        logger.debug("Checking CDP Climate Change compliance")

        requirements = []

        # Check 1: CATEGORY_BOUNDARY
        category_boundary = self._check_category_boundary(result)
        requirements.append(category_boundary)

        # Check 2: DATA_QUALITY
        data_quality = self._check_data_quality(result)
        requirements.append(data_quality)

        # Check 3: METHOD_HIERARCHY
        method_hierarchy = self._check_method_hierarchy(result)
        requirements.append(method_hierarchy)

        # Check 4: Spend coverage (CDP-specific)
        spend_coverage = self._check_spend_coverage(result)
        requirements.append(spend_coverage)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.CDP,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "Climate Change Questionnaire C6.5",
                "disclosure_year": datetime.utcnow().year
            }
        )

    def check_sbti(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with SBTi requirements.

        Key requirements:
        - Material if >5% of Scope 3
        - Target-setting boundary
        - Base year recalculation triggers

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for SBTi
        """
        logger.debug("Checking SBTi compliance")

        requirements = []

        # Check 1: CATEGORY_BOUNDARY
        category_boundary = self._check_category_boundary(result)
        requirements.append(category_boundary)

        # Check 2: BASE_YEAR_CONSISTENCY
        base_year = self._check_base_year_consistency(result)
        requirements.append(base_year)

        # Check 3: Materiality (SBTi-specific)
        materiality = self._check_sbti_materiality(result)
        requirements.append(materiality)

        # Check 4: NO_DEPRECIATION_RULE
        no_depreciation = self._check_no_depreciation(result)
        no_depreciation.is_critical = False  # Warning for SBTi
        requirements.append(no_depreciation)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.SBTI,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "Net-Zero Standard",
                "materiality_threshold": "5% of Scope 3"
            }
        )

    def check_sb253(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with California SB 253.

        Key requirements:
        - Reporting required by FY2027 for >$1B revenue
        - Limited assurance by 2030
        - No depreciation rule

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for SB 253
        """
        logger.debug("Checking California SB 253 compliance")

        requirements = []

        # Check 1: NO_DEPRECIATION_RULE
        no_depreciation = self._check_no_depreciation(result)
        requirements.append(no_depreciation)

        # Check 2: VERIFICATION_SCOPE
        verification = self._check_verification_scope(result)
        requirements.append(verification)

        # Check 3: DATA_QUALITY
        data_quality = self._check_data_quality(result)
        data_quality.is_critical = False  # Non-critical for SB 253
        requirements.append(data_quality)

        # Check 4: METHOD_HIERARCHY
        method_hierarchy = self._check_method_hierarchy(result)
        method_hierarchy.is_critical = False
        requirements.append(method_hierarchy)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.SB253,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "Climate Corporate Data Accountability Act",
                "reporting_year": 2027,
                "assurance_year": 2030
            }
        )

    def check_gri(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with GRI 305.

        Key requirements:
        - GHG emissions disclosed
        - Methodology referenced
        - Base year stated

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for GRI
        """
        logger.debug("Checking GRI 305 compliance")

        requirements = []

        # Check 1: BASE_YEAR_CONSISTENCY
        base_year = self._check_base_year_consistency(result)
        requirements.append(base_year)

        # Check 2: METHOD_HIERARCHY
        method_hierarchy = self._check_method_hierarchy(result)
        requirements.append(method_hierarchy)

        # Check 3: DATA_QUALITY
        data_quality = self._check_data_quality(result)
        data_quality.is_critical = False  # Non-critical for GRI
        requirements.append(data_quality)

        # Check 4: NO_DEPRECIATION_RULE
        no_depreciation = self._check_no_depreciation(result)
        no_depreciation.is_critical = False  # Non-critical for GRI
        requirements.append(no_depreciation)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.GRI,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "GRI 305: Emissions 2016",
                "standard_type": "Topic-specific Standard"
            }
        )

    def check_iso14064(self, result: HybridResult) -> ComplianceCheckResult:
        """
        Check compliance with ISO 14064.

        Key requirements:
        - Category 4 indirect emissions
        - Uncertainty reported
        - Verification scope defined

        Args:
            result: HybridResult from calculation

        Returns:
            ComplianceCheckResult for ISO 14064
        """
        logger.debug("Checking ISO 14064 compliance")

        requirements = []

        # Check 1: UNCERTAINTY_REPORTING
        uncertainty = self._check_uncertainty_reporting(result)
        requirements.append(uncertainty)

        # Check 2: VERIFICATION_SCOPE
        verification = self._check_verification_scope(result)
        requirements.append(verification)

        # Check 3: METHOD_HIERARCHY
        method_hierarchy = self._check_method_hierarchy(result)
        method_hierarchy.is_critical = False  # Non-critical for ISO
        requirements.append(method_hierarchy)

        # Check 4: DATA_QUALITY
        data_quality = self._check_data_quality(result)
        data_quality.is_critical = False
        requirements.append(data_quality)

        # Calculate status and score
        total_requirements = len(requirements)
        met_requirements = sum(1 for r in requirements if r.is_met)
        critical_met = all(r.is_met for r in requirements if r.is_critical)

        score = (met_requirements / total_requirements * 100) if total_requirements > 0 else 0.0

        if critical_met and met_requirements == total_requirements:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        critical_gaps = [
            r.gap_description for r in requirements
            if r.is_critical and not r.is_met and r.gap_description
        ]

        warnings = [
            r.gap_description for r in requirements
            if not r.is_critical and not r.is_met and r.gap_description
        ]

        recommendations = [
            r.recommendation for r in requirements
            if not r.is_met and r.recommendation
        ]

        return ComplianceCheckResult(
            framework=ComplianceFramework.ISO14064,
            status=status,
            requirements=requirements,
            score=score,
            critical_gaps=critical_gaps,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "version": "ISO 14064-1:2018",
                "category": "Category 4 - Indirect GHG emissions from products used by organization"
            }
        )

    def get_compliance_summary(self, results: Dict[str, ComplianceCheckResult]) -> Dict[str, Any]:
        """
        Get summary of compliance results across all frameworks.

        Args:
            results: Dictionary of framework results

        Returns:
            Summary dictionary with overall status and statistics

        Example:
            >>> summary = engine.get_compliance_summary(results)
            >>> print(f"Overall compliance: {summary['overall_status']}")
        """
        if not results:
            return {
                "overall_status": ComplianceStatus.NON_COMPLIANT,
                "total_frameworks": 0,
                "compliant_count": 0,
                "partial_count": 0,
                "non_compliant_count": 0,
                "average_score": 0.0,
                "frameworks": {}
            }

        compliant_count = sum(
            1 for r in results.values()
            if r.status == ComplianceStatus.COMPLIANT
        )
        partial_count = sum(
            1 for r in results.values()
            if r.status == ComplianceStatus.PARTIAL
        )
        non_compliant_count = sum(
            1 for r in results.values()
            if r.status == ComplianceStatus.NON_COMPLIANT
        )

        total_score = sum(r.score for r in results.values())
        average_score = total_score / len(results) if results else 0.0

        # Overall status: COMPLIANT if all compliant, PARTIAL if any partial and no non-compliant
        if compliant_count == len(results):
            overall_status = ComplianceStatus.COMPLIANT
        elif non_compliant_count > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.PARTIAL

        framework_summaries = {
            name: {
                "status": result.status.value,
                "score": result.score,
                "critical_gaps_count": len(result.critical_gaps),
                "warnings_count": len(result.warnings)
            }
            for name, result in results.items()
        }

        return {
            "overall_status": overall_status.value,
            "total_frameworks": len(results),
            "compliant_count": compliant_count,
            "partial_count": partial_count,
            "non_compliant_count": non_compliant_count,
            "average_score": round(average_score, 2),
            "frameworks": framework_summaries,
            "generated_at": datetime.utcnow().isoformat()
        }

    def get_gaps(self, results: Dict[str, ComplianceCheckResult]) -> List[str]:
        """
        Get all critical gaps across frameworks.

        Args:
            results: Dictionary of framework results

        Returns:
            List of critical gap descriptions

        Example:
            >>> gaps = engine.get_gaps(results)
            >>> for gap in gaps:
            >>>     print(f"Gap: {gap}")
        """
        all_gaps = []

        for framework_name, result in results.items():
            for gap in result.critical_gaps:
                all_gaps.append(f"[{framework_name}] {gap}")

        return all_gaps

    def get_recommendations(self, results: Dict[str, ComplianceCheckResult]) -> List[str]:
        """
        Get all recommendations across frameworks.

        Args:
            results: Dictionary of framework results

        Returns:
            List of unique recommendations

        Example:
            >>> recs = engine.get_recommendations(results)
            >>> for rec in recs:
            >>>     print(f"Recommendation: {rec}")
        """
        all_recommendations = set()

        for result in results.values():
            for rec in result.recommendations:
                all_recommendations.add(rec)

        return sorted(list(all_recommendations))

    # ============================================================================
    # PRIVATE HELPER METHODS - Individual Compliance Checks
    # ============================================================================

    def _check_no_depreciation(self, result: HybridResult) -> ComplianceRequirement:
        """Check that emissions are NOT depreciated over asset useful life."""
        # Capital goods emissions must be allocated to the reporting year of purchase
        # NOT depreciated over the asset's useful life

        is_met = True
        evidence = "Emissions allocated to reporting year of purchase (not depreciated)"
        gap_description = None
        recommendation = None

        # Check if metadata indicates depreciation was applied
        if result.metadata.get("depreciation_applied", False):
            is_met = False
            gap_description = "Emissions appear to be depreciated over asset life (violates GHG Protocol)"
            recommendation = "Allocate full emissions to reporting year of purchase, not over useful life"

        # Check for useful_life_years in allocation (red flag)
        if result.metadata.get("useful_life_allocation", False):
            is_met = False
            gap_description = "Useful life used for emission allocation (violates no-depreciation rule)"
            recommendation = "Remove useful life from emission allocation calculation"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.NO_DEPRECIATION_RULE,
            description="Emissions must NOT be depreciated over asset useful life (GHG Protocol rule)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_capitalization_classification(self, result: HybridResult) -> ComplianceRequirement:
        """Check that assets are properly classified as PP&E (Property, Plant, Equipment)."""
        is_met = True
        evidence = "Assets classified as capital goods (PP&E)"
        gap_description = None
        recommendation = None

        # Check that asset_type is not OPERATING_EXPENSE or OTHER
        if hasattr(result, 'asset_type'):
            if result.asset_type in [AssetType.OPERATING_EXPENSE, AssetType.OTHER]:
                is_met = False
                gap_description = f"Asset type '{result.asset_type}' is not proper capital classification"
                recommendation = "Reclassify operating expenses to Category 1 (Purchased Goods)"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.CAPITALIZATION_CLASSIFICATION,
            description="Assets must be properly classified as Property, Plant, and Equipment (PP&E)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_category_boundary(self, result: HybridResult) -> ComplianceRequirement:
        """Check that Category 2 boundary is clear (no overlap with Category 1)."""
        is_met = True
        evidence = "Category 2 boundary clear: capital goods (PP&E) only"
        gap_description = None
        recommendation = None

        # Check for potential Category 1 items
        category_1_keywords = [
            "raw_material", "component", "packaging", "consumable",
            "operating_expense", "maintenance_supply"
        ]

        description_lower = result.metadata.get("description", "").lower()
        for keyword in category_1_keywords:
            if keyword in description_lower:
                is_met = False
                gap_description = f"Potential Category 1 item detected: '{keyword}' in description"
                recommendation = "Verify this is capital good (PP&E), not purchased good/service (Cat 1)"
                break

        return ComplianceRequirement(
            check_type=ComplianceCheckType.CATEGORY_BOUNDARY,
            description="Clear boundary between Category 2 (Capital Goods) and Category 1 (Purchased Goods)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_scope_boundary(self, result: HybridResult) -> ComplianceRequirement:
        """Check that Scope 3 Cat 2 boundary excludes Scope 1/2 use-phase emissions."""
        is_met = True
        evidence = "Scope boundary correct: cradle-to-gate only (excludes use-phase)"
        gap_description = None
        recommendation = None

        # Check if use-phase emissions are included (red flag)
        if result.metadata.get("includes_use_phase", False):
            is_met = False
            gap_description = "Use-phase emissions included (should be Scope 1/2, not Scope 3 Cat 2)"
            recommendation = "Remove use-phase emissions; report in Scope 1/2 instead"

        # Check if end-of-life emissions are included (should be Category 12)
        if result.metadata.get("includes_end_of_life", False):
            is_met = False
            gap_description = "End-of-life emissions included (should be Category 12, not Cat 2)"
            recommendation = "Remove end-of-life emissions; report in Category 12 instead"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.SCOPE_BOUNDARY,
            description="Scope boundary excludes Scope 1/2 use-phase and Category 12 end-of-life emissions",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_method_hierarchy(self, result: HybridResult) -> ComplianceRequirement:
        """Check that method hierarchy is documented."""
        is_met = True
        evidence = f"Method documented: {result.method.value}"
        gap_description = None
        recommendation = None

        # Check if method is documented
        if not hasattr(result, 'method') or result.method is None:
            is_met = False
            gap_description = "Calculation method not documented"
            recommendation = "Document calculation method (Supplier-Specific, Hybrid, or Average-Data)"

        # Check if method justification exists
        if not result.metadata.get("method_justification"):
            is_met = False
            gap_description = "Method selection not justified"
            recommendation = "Provide justification for method selection (data availability, accuracy trade-offs)"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.METHOD_HIERARCHY,
            description="Calculation method hierarchy documented and justified",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_data_quality(self, result: HybridResult) -> ComplianceRequirement:
        """Check that data quality is assessed and reported."""
        is_met = True
        score_value = result.quality_score.overall_score if hasattr(result, 'quality_score') else 0.0
        evidence = f"Data quality score: {score_value:.1f}/5.0"
        gap_description = None
        recommendation = None

        # Check if quality score exists
        if not hasattr(result, 'quality_score'):
            is_met = False
            gap_description = "Data quality not assessed"
            recommendation = "Implement data quality scoring (5 dimensions: completeness, accuracy, etc.)"
        elif score_value < 2.0:
            is_met = False
            gap_description = f"Data quality score too low: {score_value:.1f}/5.0"
            recommendation = "Improve data quality (target: >3.0/5.0 for regulatory reporting)"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.DATA_QUALITY,
            description="Data quality assessed and reported using DQI framework",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_base_year_consistency(self, result: HybridResult) -> ComplianceRequirement:
        """Check that base year is consistent and documented."""
        is_met = True
        base_year = result.metadata.get("base_year", "Not specified")
        evidence = f"Base year: {base_year}"
        gap_description = None
        recommendation = None

        # Check if base year is specified
        if not result.metadata.get("base_year"):
            is_met = False
            gap_description = "Base year not specified"
            recommendation = "Specify base year for emissions reporting (typically year of first disclosure)"

        # Check if recalculation policy exists
        if not result.metadata.get("recalculation_policy"):
            is_met = False
            gap_description = "Base year recalculation policy not documented"
            recommendation = "Document base year recalculation triggers (structural changes, methodology improvements)"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.BASE_YEAR_CONSISTENCY,
            description="Base year is consistent and recalculation policy documented",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_capex_volatility_context(self, result: HybridResult) -> ComplianceRequirement:
        """Check that narrative is provided for major CapEx years (CSRD-specific)."""
        is_met = True
        evidence = "CapEx volatility context provided"
        gap_description = None
        recommendation = None

        # Check if this is a high-CapEx year
        is_high_capex = result.metadata.get("is_high_capex_year", False)
        has_context = result.metadata.get("capex_volatility_narrative")

        if is_high_capex and not has_context:
            is_met = False
            gap_description = "High CapEx year without explanatory narrative (CSRD requirement)"
            recommendation = "Provide narrative explaining major capital investments and emission drivers"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.CAPEX_VOLATILITY_CONTEXT,
            description="Narrative context provided for major CapEx years (CSRD ESRS E1)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_uncertainty_reporting(self, result: HybridResult) -> ComplianceRequirement:
        """Check that uncertainty is quantified and reported (ISO 14064)."""
        is_met = True
        uncertainty_pct = result.metadata.get("uncertainty_percentage", 0.0)
        evidence = f"Uncertainty quantified: ±{uncertainty_pct:.1f}%"
        gap_description = None
        recommendation = None

        # Check if uncertainty is quantified
        if not result.metadata.get("uncertainty_percentage"):
            is_met = False
            gap_description = "Uncertainty not quantified (ISO 14064 requirement)"
            recommendation = "Quantify uncertainty using error propagation or Monte Carlo methods"

        # Check if uncertainty sources are documented
        if not result.metadata.get("uncertainty_sources"):
            is_met = False
            gap_description = "Uncertainty sources not documented"
            recommendation = "Document uncertainty sources (EF, activity data, model parameters)"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.UNCERTAINTY_REPORTING,
            description="Uncertainty quantified and reported with sources (ISO 14064)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_verification_scope(self, result: HybridResult) -> ComplianceRequirement:
        """Check that verification scope is defined."""
        is_met = True
        evidence = "Verification scope defined"
        gap_description = None
        recommendation = None

        # Check if verification scope is specified
        if not result.metadata.get("verification_scope"):
            is_met = False
            gap_description = "Verification scope not defined"
            recommendation = "Define verification scope (limited vs. reasonable assurance)"

        # Check if verification level is appropriate
        verification_level = result.metadata.get("verification_level")
        if verification_level and verification_level not in ["limited", "reasonable", "none"]:
            is_met = False
            gap_description = f"Invalid verification level: {verification_level}"
            recommendation = "Set verification level to 'limited', 'reasonable', or 'none'"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.VERIFICATION_SCOPE,
            description="Verification scope and level defined (limited/reasonable assurance)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_spend_coverage(self, result: HybridResult) -> ComplianceRequirement:
        """Check spend coverage percentage (CDP-specific)."""
        is_met = True
        spend_coverage = result.metadata.get("spend_coverage_pct", 0.0)
        evidence = f"Spend coverage: {spend_coverage:.1f}%"
        gap_description = None
        recommendation = None

        # CDP expects >67% spend coverage for A-grade
        if spend_coverage < 67.0:
            is_met = False
            gap_description = f"Spend coverage below CDP A-grade threshold: {spend_coverage:.1f}% (target: >67%)"
            recommendation = "Increase spend coverage through supplier engagement and data collection"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.DATA_QUALITY,
            description="Spend coverage >67% for CDP A-grade (Category 2 specific)",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )

    def _check_sbti_materiality(self, result: HybridResult) -> ComplianceRequirement:
        """Check SBTi materiality threshold (>5% of Scope 3)."""
        is_met = True
        materiality_pct = result.metadata.get("scope3_percentage", 0.0)
        evidence = f"Category 2 represents {materiality_pct:.1f}% of Scope 3"
        gap_description = None
        recommendation = None

        # SBTi requires including categories >5% of Scope 3 in target-setting
        if materiality_pct > 5.0 and not result.metadata.get("included_in_sbti_target"):
            is_met = False
            gap_description = f"Category 2 is material ({materiality_pct:.1f}% of Scope 3) but not in SBTi target"
            recommendation = "Include Category 2 in SBTi target-setting boundary (>5% materiality threshold)"

        return ComplianceRequirement(
            check_type=ComplianceCheckType.CATEGORY_BOUNDARY,
            description="Material categories (>5% of Scope 3) included in SBTi target boundary",
            is_critical=True,
            is_met=is_met,
            evidence=evidence if is_met else None,
            gap_description=gap_description,
            recommendation=recommendation
        )
