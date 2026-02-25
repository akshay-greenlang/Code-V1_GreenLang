"""
ComplianceCheckerEngine - Engine 6/7 for AGENT-MRV-014 (Purchased Goods & Services).

This module implements regulatory compliance validation for Category 1 emissions
across 7 frameworks: GHG Protocol, CSRD/ESRS E1, CDP, SBTi, SB 253, GRI 305, ISO 14064.

Zero-Hallucination Approach:
- Validates against explicit framework requirements
- Checks coverage thresholds, DQI scores, methodology disclosures
- All validation logic deterministic (no LLM)
- Framework requirements sourced from official standards

Example:
    >>> engine = ComplianceCheckerEngine()
    >>> check_results = engine.check_all_frameworks(result, disclosures)
    >>> summary = engine.get_compliance_summary(check_results)
    >>> assert summary["compliant_count"] >= 0
"""

from typing import Dict, List, Optional, Any, Set
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import logging
import threading

from greenlang.purchased_goods_services.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    ZERO,
    ONE,
    ONE_HUNDRED,
    DECIMAL_PLACES,
    CalculationMethod,
    ComplianceFramework,
    ComplianceStatus,
    CoverageLevel,
    DQIDimension,
    DQIScore,
    COVERAGE_THRESHOLDS,
    FRAMEWORK_REQUIRED_DISCLOSURES,
    DQI_SCORE_VALUES,
    DQI_QUALITY_TIERS,
    HybridResult,
    ComplianceRequirement,
    ComplianceCheckResult,
    DQIAssessment,
    CoverageReport,
)
from greenlang.purchased_goods_services.config import PurchasedGoodsServicesConfig
from greenlang.purchased_goods_services.metrics import PurchasedGoodsServicesMetrics
from greenlang.purchased_goods_services.provenance import PurchasedGoodsProvenanceTracker

__all__ = [
    "ComplianceCheckerEngine",
]

logger = logging.getLogger(__name__)


class ComplianceCheckerEngine:
    """
    Compliance validation engine for Category 1 emissions.

    This engine validates emissions calculations and disclosures against
    7 major regulatory frameworks. Each framework has specific requirements
    for methodology, coverage, data quality, and reporting.

    Thread-safe singleton with comprehensive validation logic.

    Attributes:
        config: Agent configuration
        metrics: Performance metrics tracker
        provenance: Audit trail tracker
        _framework_validators: Map of framework-specific validation methods

    Example:
        >>> engine = ComplianceCheckerEngine()
        >>> result = HybridResult(...)
        >>> disclosures = {"methodology": "hybrid", "ef_sources": [...]}
        >>> checks = engine.check_all_frameworks(result, disclosures)
        >>> for check in checks:
        ...     if check.status == ComplianceStatus.NON_COMPLIANT:
        ...         print(f"{check.framework}: {check.gaps}")
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls) -> "ComplianceCheckerEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize compliance checker engine."""
        if hasattr(self, "_initialized"):
            return

        self.config = PurchasedGoodsServicesConfig()
        self.metrics = PurchasedGoodsServicesMetrics()
        self.provenance = PurchasedGoodsProvenanceTracker()

        # Framework-specific validation methods
        self._framework_validators = {
            ComplianceFramework.GHG_PROTOCOL: self._check_ghg_protocol,
            ComplianceFramework.CSRD_ESRS_E1: self._check_csrd_esrs,
            ComplianceFramework.CDP: self._check_cdp,
            ComplianceFramework.SBTI: self._check_sbti,
            ComplianceFramework.SB_253: self._check_sb253,
            ComplianceFramework.GRI_305: self._check_gri,
            ComplianceFramework.ISO_14064: self._check_iso_14064,
        }

        self._initialized = True
        logger.info(f"{AGENT_ID} v{VERSION}: ComplianceCheckerEngine initialized")

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def check_all_frameworks(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
        frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> List[ComplianceCheckResult]:
        """
        Check compliance across all or specified frameworks.

        Args:
            result: Calculation result to validate
            disclosures: Additional disclosure information
            frameworks: Specific frameworks to check (default: all 7)

        Returns:
            List of compliance check results, one per framework

        Example:
            >>> result = HybridResult(...)
            >>> disclosures = {"methodology": "hybrid"}
            >>> checks = engine.check_all_frameworks(result, disclosures)
            >>> compliant = [c for c in checks if c.status == ComplianceStatus.COMPLIANT]
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Default to all frameworks if not specified
            if frameworks is None:
                frameworks = list(ComplianceFramework)

            # Validate each framework
            check_results = []
            for framework in frameworks:
                check_result = self.check_framework(framework, result, disclosures)
                check_results.append(check_result)

            # Record metrics
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.metrics.record_operation(
                "compliance_check_all",
                processing_time_ms,
                {"framework_count": len(frameworks)},
            )

            # Record provenance
            self.provenance.record_event(
                "compliance_check_all_frameworks",
                {
                    "frameworks": [f.value for f in frameworks],
                    "check_count": len(check_results),
                    "compliant_count": sum(
                        1 for c in check_results if c.status == ComplianceStatus.COMPLIANT
                    ),
                },
            )

            logger.info(
                f"Compliance check completed: {len(frameworks)} frameworks, "
                f"{sum(1 for c in check_results if c.status == ComplianceStatus.COMPLIANT)} compliant"
            )

            return check_results

        except Exception as e:
            logger.error(f"Compliance check failed: {e}", exc_info=True)
            self.metrics.record_error("compliance_check_all", str(e))
            raise

    def check_framework(
        self,
        framework: ComplianceFramework,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Check compliance for a specific framework.

        Args:
            framework: Framework to validate against
            result: Calculation result
            disclosures: Additional disclosures

        Returns:
            Compliance check result with status and gaps

        Raises:
            ValueError: If framework validator not implemented
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Get framework-specific validator
            validator = self._framework_validators.get(framework)
            if validator is None:
                raise ValueError(f"No validator implemented for {framework.value}")

            # Run validation
            check_result = validator(result, disclosures)

            # Record metrics
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.metrics.record_operation(
                f"compliance_check_{framework.value}",
                processing_time_ms,
                {"status": check_result.status.value},
            )

            return check_result

        except Exception as e:
            logger.error(f"Framework check failed for {framework.value}: {e}", exc_info=True)
            self.metrics.record_error(f"compliance_check_{framework.value}", str(e))
            raise

    def _check_ghg_protocol(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against GHG Protocol Corporate Standard (11 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. methodology_disclosed
        3. emission_factor_sources
        4. data_quality_indicators
        5. coverage_percentage >= 80%
        6. organizational_boundary
        7. double_counting_avoided
        8. base_year_emissions
        9. exclusions_documented
        10. gwp_values_disclosed
        11. uncertainty_quantified

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.GHG_PROTOCOL
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="GHG-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Methodology disclosed
        methodology = disclosures.get("methodology")
        req = ComplianceRequirement(
            requirement_id="GHG-2",
            description="Methodology disclosed",
            is_met=methodology is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Calculation methodology not disclosed")

        # Requirement 3: EF sources documented
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="GHG-3",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources not documented")

        # Requirement 4: DQI provided
        dqi_provided = result.data_quality_score is not None
        req = ComplianceRequirement(
            requirement_id="GHG-4",
            description="Data quality indicators provided",
            is_met=dqi_provided,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Data quality indicators missing")

        # Requirement 5: Coverage >= 80%
        coverage_threshold = COVERAGE_THRESHOLDS.get(framework, Decimal("0.80"))
        coverage_met = result.coverage_percentage >= coverage_threshold * ONE_HUNDRED
        req = ComplianceRequirement(
            requirement_id="GHG-5",
            description=f"Coverage >= {coverage_threshold * 100}%",
            is_met=coverage_met,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append(
                f"Coverage {result.coverage_percentage}% below threshold {coverage_threshold * 100}%"
            )

        # Requirement 6: Organizational boundary
        boundary = disclosures.get("organizational_boundary")
        req = ComplianceRequirement(
            requirement_id="GHG-6",
            description="Organizational boundary defined",
            is_met=boundary is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Organizational boundary not defined")

        # Requirement 7: Double-counting avoided
        double_counting = disclosures.get("double_counting_avoided", False)
        req = ComplianceRequirement(
            requirement_id="GHG-7",
            description="Double-counting avoided",
            is_met=double_counting,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Double-counting avoidance not confirmed")

        # Requirement 8: Base year emissions
        base_year = disclosures.get("base_year_emissions")
        req = ComplianceRequirement(
            requirement_id="GHG-8",
            description="Base year emissions provided",
            is_met=base_year is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Base year emissions not provided")

        # Requirement 9: Exclusions documented
        exclusions = disclosures.get("exclusions_documented", False)
        req = ComplianceRequirement(
            requirement_id="GHG-9",
            description="Exclusions documented",
            is_met=exclusions,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Exclusions not documented")

        # Requirement 10: GWP values disclosed
        gwp_disclosed = disclosures.get("gwp_values_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="GHG-10",
            description="GWP values disclosed",
            is_met=gwp_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("GWP values not disclosed")

        # Requirement 11: Uncertainty quantified
        uncertainty = result.uncertainty_percentage
        req = ComplianceRequirement(
            requirement_id="GHG-11",
            description="Uncertainty quantified",
            is_met=uncertainty is not None and uncertainty > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Uncertainty not quantified")

        # Determine overall status
        critical_met = all(r.is_met for r in requirements[:6])  # First 6 critical
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def _check_csrd_esrs(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against CSRD/ESRS E1 (12 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. methodology_disclosed
        3. emission_factor_sources
        4. data_quality_indicators
        5. coverage_percentage_disclosed
        6. supplier_engagement_disclosed
        7. emission_intensity_metrics
        8. year_over_year_changes
        9. base_year_disclosed
        10. reduction_targets
        11. organizational_boundary
        12. gwp_ar6_disclosed

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.CSRD_ESRS_E1
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="ESRS-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Methodology
        methodology = disclosures.get("methodology")
        req = ComplianceRequirement(
            requirement_id="ESRS-2",
            description="Methodology disclosed",
            is_met=methodology is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Methodology not disclosed")

        # Requirement 3: EF sources
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="ESRS-3",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources missing")

        # Requirement 4: DQI
        dqi_provided = result.data_quality_score is not None
        req = ComplianceRequirement(
            requirement_id="ESRS-4",
            description="Data quality indicators provided",
            is_met=dqi_provided,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Data quality indicators missing")

        # Requirement 5: Coverage disclosed
        coverage_disclosed = result.coverage_percentage > ZERO
        req = ComplianceRequirement(
            requirement_id="ESRS-5",
            description="Coverage percentage disclosed",
            is_met=coverage_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Coverage percentage not disclosed")

        # Requirement 6: Supplier engagement
        supplier_engagement = disclosures.get("supplier_engagement_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="ESRS-6",
            description="Supplier engagement disclosed",
            is_met=supplier_engagement,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Supplier engagement not disclosed")

        # Requirement 7: Emission intensity
        intensity_metrics = disclosures.get("emission_intensity_metrics", [])
        req = ComplianceRequirement(
            requirement_id="ESRS-7",
            description="Emission intensity metrics provided",
            is_met=len(intensity_metrics) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Emission intensity metrics missing")

        # Requirement 8: Year-over-year changes
        yoy_changes = disclosures.get("year_over_year_changes")
        req = ComplianceRequirement(
            requirement_id="ESRS-8",
            description="Year-over-year changes disclosed",
            is_met=yoy_changes is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Year-over-year changes not disclosed")

        # Requirement 9: Base year
        base_year = disclosures.get("base_year_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="ESRS-9",
            description="Base year disclosed",
            is_met=base_year,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Base year not disclosed")

        # Requirement 10: Reduction targets
        targets = disclosures.get("reduction_targets", [])
        req = ComplianceRequirement(
            requirement_id="ESRS-10",
            description="Reduction targets disclosed",
            is_met=len(targets) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Reduction targets not disclosed")

        # Requirement 11: Organizational boundary
        boundary = disclosures.get("organizational_boundary")
        req = ComplianceRequirement(
            requirement_id="ESRS-11",
            description="Organizational boundary defined",
            is_met=boundary is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Organizational boundary not defined")

        # Requirement 12: GWP AR6
        gwp_ar6 = disclosures.get("gwp_ar6_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="ESRS-12",
            description="GWP AR6 values disclosed",
            is_met=gwp_ar6,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("GWP AR6 values not disclosed")

        # Determine status
        critical_met = all(r.is_met for r in requirements[:5] + [requirements[8], requirements[10]])
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def _check_cdp(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against CDP Climate Change (10 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. relevance_assessment
        3. methodology_disclosed
        4. primary_data_percentage
        5. secondary_data_percentage
        6. emission_factor_sources
        7. coverage_percentage
        8. supplier_scoring_disclosed
        9. verification_status
        10. organizational_boundary

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.CDP
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="CDP-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Relevance assessment
        relevance = disclosures.get("relevance_assessment")
        req = ComplianceRequirement(
            requirement_id="CDP-2",
            description="Relevance assessment completed",
            is_met=relevance is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Relevance assessment missing")

        # Requirement 3: Methodology
        methodology = disclosures.get("methodology")
        req = ComplianceRequirement(
            requirement_id="CDP-3",
            description="Methodology disclosed",
            is_met=methodology is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Methodology not disclosed")

        # Requirement 4: Primary data %
        primary_pct = disclosures.get("primary_data_percentage")
        req = ComplianceRequirement(
            requirement_id="CDP-4",
            description="Primary data percentage disclosed",
            is_met=primary_pct is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Primary data percentage not disclosed")

        # Requirement 5: Secondary data %
        secondary_pct = disclosures.get("secondary_data_percentage")
        req = ComplianceRequirement(
            requirement_id="CDP-5",
            description="Secondary data percentage disclosed",
            is_met=secondary_pct is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Secondary data percentage not disclosed")

        # Requirement 6: EF sources
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="CDP-6",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources missing")

        # Requirement 7: Coverage
        coverage_disclosed = result.coverage_percentage > ZERO
        req = ComplianceRequirement(
            requirement_id="CDP-7",
            description="Coverage percentage disclosed",
            is_met=coverage_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Coverage percentage not disclosed")

        # Requirement 8: Supplier scoring
        supplier_scoring = disclosures.get("supplier_scoring_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="CDP-8",
            description="Supplier scoring disclosed",
            is_met=supplier_scoring,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Supplier scoring not disclosed")

        # Requirement 9: Verification
        verification = disclosures.get("verification_status")
        req = ComplianceRequirement(
            requirement_id="CDP-9",
            description="Verification status disclosed",
            is_met=verification is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Verification status not disclosed")

        # Requirement 10: Organizational boundary
        boundary = disclosures.get("organizational_boundary")
        req = ComplianceRequirement(
            requirement_id="CDP-10",
            description="Organizational boundary defined",
            is_met=boundary is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Organizational boundary not defined")

        # Determine status
        critical_met = all(r.is_met for r in [requirements[0], requirements[1], requirements[2], requirements[5], requirements[6], requirements[9]])
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def _check_sbti(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against Science Based Targets initiative (8 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. base_year_total_emissions
        3. target_year
        4. reduction_percentage
        5. coverage_percentage >= 67%
        6. supplier_targets_disclosed
        7. methodology_disclosed
        8. emission_factor_sources

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.SBTI
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="SBTI-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Base year total
        base_year_total = disclosures.get("base_year_total_emissions")
        req = ComplianceRequirement(
            requirement_id="SBTI-2",
            description="Base year total emissions provided",
            is_met=base_year_total is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Base year total emissions missing")

        # Requirement 3: Target year
        target_year = disclosures.get("target_year")
        req = ComplianceRequirement(
            requirement_id="SBTI-3",
            description="Target year specified",
            is_met=target_year is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Target year not specified")

        # Requirement 4: Reduction %
        reduction_pct = disclosures.get("reduction_percentage")
        req = ComplianceRequirement(
            requirement_id="SBTI-4",
            description="Reduction percentage specified",
            is_met=reduction_pct is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Reduction percentage not specified")

        # Requirement 5: Coverage >= 67%
        coverage_threshold = COVERAGE_THRESHOLDS.get(framework, Decimal("0.67"))
        coverage_met = result.coverage_percentage >= coverage_threshold * ONE_HUNDRED
        req = ComplianceRequirement(
            requirement_id="SBTI-5",
            description=f"Coverage >= {coverage_threshold * 100}%",
            is_met=coverage_met,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append(
                f"Coverage {result.coverage_percentage}% below threshold {coverage_threshold * 100}%"
            )

        # Requirement 6: Supplier targets
        supplier_targets = disclosures.get("supplier_targets_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="SBTI-6",
            description="Supplier targets disclosed",
            is_met=supplier_targets,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Supplier targets not disclosed")

        # Requirement 7: Methodology
        methodology = disclosures.get("methodology")
        req = ComplianceRequirement(
            requirement_id="SBTI-7",
            description="Methodology disclosed",
            is_met=methodology is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Methodology not disclosed")

        # Requirement 8: EF sources
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="SBTI-8",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources missing")

        # Determine status - all requirements critical for SBTi
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif requirements[0].is_met and requirements[4].is_met:  # Total + coverage
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def _check_sb253(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against California SB 253 (8 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. methodology_disclosed
        3. emission_factor_sources
        4. data_quality_indicators
        5. coverage_percentage
        6. assurance_level
        7. reporting_entity_revenue
        8. gwp_values_disclosed

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.SB_253
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="SB253-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Methodology
        methodology = disclosures.get("methodology")
        req = ComplianceRequirement(
            requirement_id="SB253-2",
            description="Methodology disclosed",
            is_met=methodology is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Methodology not disclosed")

        # Requirement 3: EF sources
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="SB253-3",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources missing")

        # Requirement 4: DQI
        dqi_provided = result.data_quality_score is not None
        req = ComplianceRequirement(
            requirement_id="SB253-4",
            description="Data quality indicators provided",
            is_met=dqi_provided,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Data quality indicators missing")

        # Requirement 5: Coverage
        coverage_disclosed = result.coverage_percentage > ZERO
        req = ComplianceRequirement(
            requirement_id="SB253-5",
            description="Coverage percentage disclosed",
            is_met=coverage_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Coverage percentage not disclosed")

        # Requirement 6: Assurance level
        assurance = disclosures.get("assurance_level")
        req = ComplianceRequirement(
            requirement_id="SB253-6",
            description="Assurance level disclosed",
            is_met=assurance is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Assurance level not disclosed")

        # Requirement 7: Entity revenue
        revenue = disclosures.get("reporting_entity_revenue")
        req = ComplianceRequirement(
            requirement_id="SB253-7",
            description="Reporting entity revenue disclosed",
            is_met=revenue is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Reporting entity revenue not disclosed")

        # Requirement 8: GWP values
        gwp_disclosed = disclosures.get("gwp_values_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="SB253-8",
            description="GWP values disclosed",
            is_met=gwp_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("GWP values not disclosed")

        # Determine status
        critical_met = all(r.is_met for r in requirements[:6])
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def _check_gri(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against GRI 305 (8 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. methodology_disclosed
        3. emission_factor_sources
        4. gwp_values_disclosed
        5. consolidation_approach
        6. base_year
        7. standards_frameworks
        8. significant_changes

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.GRI_305
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="GRI-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Methodology
        methodology = disclosures.get("methodology")
        req = ComplianceRequirement(
            requirement_id="GRI-2",
            description="Methodology disclosed",
            is_met=methodology is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Methodology not disclosed")

        # Requirement 3: EF sources
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="GRI-3",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources missing")

        # Requirement 4: GWP values
        gwp_disclosed = disclosures.get("gwp_values_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="GRI-4",
            description="GWP values disclosed",
            is_met=gwp_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("GWP values not disclosed")

        # Requirement 5: Consolidation approach
        consolidation = disclosures.get("consolidation_approach")
        req = ComplianceRequirement(
            requirement_id="GRI-5",
            description="Consolidation approach disclosed",
            is_met=consolidation is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Consolidation approach not disclosed")

        # Requirement 6: Base year
        base_year = disclosures.get("base_year")
        req = ComplianceRequirement(
            requirement_id="GRI-6",
            description="Base year disclosed",
            is_met=base_year is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Base year not disclosed")

        # Requirement 7: Standards/frameworks
        standards = disclosures.get("standards_frameworks", [])
        req = ComplianceRequirement(
            requirement_id="GRI-7",
            description="Standards/frameworks disclosed",
            is_met=len(standards) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Standards/frameworks not disclosed")

        # Requirement 8: Significant changes
        changes = disclosures.get("significant_changes")
        req = ComplianceRequirement(
            requirement_id="GRI-8",
            description="Significant changes disclosed",
            is_met=changes is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Significant changes not disclosed")

        # Determine status
        critical_met = all(r.is_met for r in requirements[:5])
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def _check_iso_14064(
        self,
        result: HybridResult,
        disclosures: Dict[str, Any],
    ) -> ComplianceCheckResult:
        """
        Validate against ISO 14064-1 (12 requirements).

        Requirements:
        1. total_emissions_kg_co2e
        2. method_justification
        3. co2_emissions_by_gas
        4. ch4_emissions_by_gas
        5. n2o_emissions_by_gas
        6. emission_factor_sources
        7. gwp_values_disclosed
        8. uncertainty_quantified
        9. organizational_boundary
        10. reporting_period
        11. base_year
        12. data_quality_indicators

        Args:
            result: Calculation result
            disclosures: Framework disclosures

        Returns:
            Compliance check result
        """
        framework = ComplianceFramework.ISO_14064
        requirements = []
        gaps = []
        warnings = []

        # Requirement 1: Total emissions
        req = ComplianceRequirement(
            requirement_id="ISO-1",
            description="Total emissions calculated",
            is_met=result.total_emissions_kg_co2e > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Total emissions not calculated")

        # Requirement 2: Method justification
        method_justification = disclosures.get("method_justification")
        req = ComplianceRequirement(
            requirement_id="ISO-2",
            description="Method justification provided",
            is_met=method_justification is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Method justification missing")

        # Requirement 3: CO2 by gas
        co2_by_gas = disclosures.get("co2_emissions_by_gas")
        req = ComplianceRequirement(
            requirement_id="ISO-3",
            description="CO2 emissions by gas disclosed",
            is_met=co2_by_gas is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("CO2 emissions by gas not disclosed")

        # Requirement 4: CH4 by gas
        ch4_by_gas = disclosures.get("ch4_emissions_by_gas")
        req = ComplianceRequirement(
            requirement_id="ISO-4",
            description="CH4 emissions by gas disclosed",
            is_met=ch4_by_gas is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("CH4 emissions by gas not disclosed")

        # Requirement 5: N2O by gas
        n2o_by_gas = disclosures.get("n2o_emissions_by_gas")
        req = ComplianceRequirement(
            requirement_id="ISO-5",
            description="N2O emissions by gas disclosed",
            is_met=n2o_by_gas is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("N2O emissions by gas not disclosed")

        # Requirement 6: EF sources
        ef_sources = disclosures.get("ef_sources", [])
        req = ComplianceRequirement(
            requirement_id="ISO-6",
            description="Emission factor sources documented",
            is_met=len(ef_sources) > 0,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Emission factor sources missing")

        # Requirement 7: GWP values
        gwp_disclosed = disclosures.get("gwp_values_disclosed", False)
        req = ComplianceRequirement(
            requirement_id="ISO-7",
            description="GWP values disclosed",
            is_met=gwp_disclosed,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("GWP values not disclosed")

        # Requirement 8: Uncertainty
        uncertainty = result.uncertainty_percentage
        req = ComplianceRequirement(
            requirement_id="ISO-8",
            description="Uncertainty quantified",
            is_met=uncertainty is not None and uncertainty > ZERO,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Uncertainty not quantified")

        # Requirement 9: Organizational boundary
        boundary = disclosures.get("organizational_boundary")
        req = ComplianceRequirement(
            requirement_id="ISO-9",
            description="Organizational boundary defined",
            is_met=boundary is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Organizational boundary not defined")

        # Requirement 10: Reporting period
        period = disclosures.get("reporting_period")
        req = ComplianceRequirement(
            requirement_id="ISO-10",
            description="Reporting period disclosed",
            is_met=period is not None,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Reporting period not disclosed")

        # Requirement 11: Base year
        base_year = disclosures.get("base_year")
        req = ComplianceRequirement(
            requirement_id="ISO-11",
            description="Base year disclosed",
            is_met=base_year is not None,
        )
        requirements.append(req)
        if not req.is_met:
            warnings.append("Base year not disclosed")

        # Requirement 12: DQI
        dqi_provided = result.data_quality_score is not None
        req = ComplianceRequirement(
            requirement_id="ISO-12",
            description="Data quality indicators provided",
            is_met=dqi_provided,
        )
        requirements.append(req)
        if not req.is_met:
            gaps.append("Data quality indicators missing")

        # Determine status
        critical_met = all(r.is_met for r in [requirements[0], requirements[1], requirements[5], requirements[6], requirements[7], requirements[8], requirements[9], requirements[11]])
        all_met = all(r.is_met for r in requirements)

        if all_met:
            status = ComplianceStatus.COMPLIANT
        elif critical_met:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            requirements=requirements,
            gaps=gaps,
            warnings=warnings,
            check_timestamp=datetime.now(timezone.utc),
        )

    def get_required_disclosures(self, framework: ComplianceFramework) -> List[str]:
        """
        Get list of required disclosures for a framework.

        Args:
            framework: Framework to get disclosures for

        Returns:
            List of required disclosure field names

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> disclosures = engine.get_required_disclosures(ComplianceFramework.GHG_PROTOCOL)
            >>> assert "methodology" in disclosures
        """
        return FRAMEWORK_REQUIRED_DISCLOSURES.get(framework, [])

    def check_coverage_threshold(
        self, coverage_pct: Decimal, framework: ComplianceFramework
    ) -> bool:
        """
        Check if coverage meets framework threshold.

        Args:
            coverage_pct: Coverage percentage (0-100)
            framework: Framework to check against

        Returns:
            True if coverage meets threshold

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> assert engine.check_coverage_threshold(Decimal("85"), ComplianceFramework.GHG_PROTOCOL)
            >>> assert not engine.check_coverage_threshold(Decimal("60"), ComplianceFramework.SBTI)
        """
        threshold = COVERAGE_THRESHOLDS.get(framework, ZERO)
        return coverage_pct >= threshold * ONE_HUNDRED

    def check_dqi_threshold(
        self, weighted_dqi: Decimal, framework: ComplianceFramework
    ) -> bool:
        """
        Check if DQI score meets framework requirements.

        Args:
            weighted_dqi: Weighted DQI score (1-5)
            framework: Framework to check against

        Returns:
            True if DQI meets threshold

        Note:
            Most frameworks require DQI >= 3.0 (Fair quality)
            SBTi and CSRD prefer DQI >= 4.0 (Good quality)

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> assert engine.check_dqi_threshold(Decimal("3.5"), ComplianceFramework.GHG_PROTOCOL)
            >>> assert not engine.check_dqi_threshold(Decimal("2.5"), ComplianceFramework.CSRD_ESRS_E1)
        """
        # Most frameworks require Fair quality (3.0) minimum
        base_threshold = Decimal("3.0")

        # SBTi and CSRD prefer Good quality (4.0)
        if framework in [ComplianceFramework.SBTI, ComplianceFramework.CSRD_ESRS_E1]:
            base_threshold = Decimal("4.0")

        return weighted_dqi >= base_threshold

    def generate_recommendations(
        self, check_results: List[ComplianceCheckResult]
    ) -> List[str]:
        """
        Generate improvement recommendations based on check results.

        Args:
            check_results: List of compliance check results

        Returns:
            List of actionable recommendations

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> checks = engine.check_all_frameworks(result, disclosures)
            >>> recommendations = engine.generate_recommendations(checks)
            >>> for rec in recommendations:
            ...     print(f"- {rec}")
        """
        recommendations = []
        recommendation_set: Set[str] = set()  # Avoid duplicates

        for check in check_results:
            # Coverage improvements
            if any("coverage" in gap.lower() for gap in check.gaps):
                rec = "Increase coverage by engaging more suppliers or collecting additional spend data"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # DQI improvements
            if any("quality" in gap.lower() for gap in check.gaps):
                rec = "Improve data quality by collecting primary supplier data instead of using secondary estimates"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # Methodology documentation
            if any("methodology" in gap.lower() for gap in check.gaps):
                rec = "Document calculation methodology including hybrid approach, allocation methods, and data sources"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # Emission factor sources
            if any("emission factor" in gap.lower() or "ef sources" in gap.lower() for gap in check.gaps):
                rec = "Document all emission factor sources including database names, versions, and geographic regions"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # Base year
            if any("base year" in gap.lower() for gap in check.gaps):
                rec = "Establish and document base year emissions for tracking progress over time"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # Uncertainty
            if any("uncertainty" in gap.lower() for gap in check.gaps):
                rec = "Quantify uncertainty using Monte Carlo simulation or IPCC uncertainty propagation methods"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # Supplier engagement
            if any("supplier" in gap.lower() for gap in check.gaps):
                rec = "Engage suppliers to collect primary emissions data and set reduction targets"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

            # Organizational boundary
            if any("boundary" in gap.lower() for gap in check.gaps):
                rec = "Define and document organizational boundary (operational control, financial control, or equity share)"
                if rec not in recommendation_set:
                    recommendations.append(rec)
                    recommendation_set.add(rec)

        # General recommendations if multiple frameworks non-compliant
        non_compliant_count = sum(
            1 for c in check_results if c.status == ComplianceStatus.NON_COMPLIANT
        )
        if non_compliant_count >= 3:
            rec = "Consider engaging third-party verification to identify and address compliance gaps"
            if rec not in recommendation_set:
                recommendations.append(rec)
                recommendation_set.add(rec)

        return recommendations

    def get_compliance_summary(
        self, check_results: List[ComplianceCheckResult]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for compliance checks.

        Args:
            check_results: List of compliance check results

        Returns:
            Dictionary with summary statistics

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> checks = engine.check_all_frameworks(result, disclosures)
            >>> summary = engine.get_compliance_summary(checks)
            >>> print(f"Compliant: {summary['compliant_count']}/{summary['total_count']}")
        """
        total_count = len(check_results)
        compliant_count = sum(
            1 for c in check_results if c.status == ComplianceStatus.COMPLIANT
        )
        partial_count = sum(
            1 for c in check_results if c.status == ComplianceStatus.PARTIAL
        )
        non_compliant_count = sum(
            1 for c in check_results if c.status == ComplianceStatus.NON_COMPLIANT
        )

        # Calculate overall compliance percentage
        compliance_pct = (
            (compliant_count / total_count * ONE_HUNDRED)
            if total_count > 0
            else ZERO
        )

        # Aggregate all gaps and warnings
        all_gaps = []
        all_warnings = []
        for check in check_results:
            all_gaps.extend(check.gaps)
            all_warnings.extend(check.warnings)

        # Generate recommendations
        recommendations = self.generate_recommendations(check_results)

        return {
            "total_count": total_count,
            "compliant_count": compliant_count,
            "partial_count": partial_count,
            "non_compliant_count": non_compliant_count,
            "compliance_percentage": float(
                compliance_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            ),
            "total_gaps": len(all_gaps),
            "total_warnings": len(all_warnings),
            "recommendations": recommendations,
            "frameworks_compliant": [
                c.framework.value
                for c in check_results
                if c.status == ComplianceStatus.COMPLIANT
            ],
            "frameworks_partial": [
                c.framework.value
                for c in check_results
                if c.status == ComplianceStatus.PARTIAL
            ],
            "frameworks_non_compliant": [
                c.framework.value
                for c in check_results
                if c.status == ComplianceStatus.NON_COMPLIANT
            ],
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on compliance checker engine.

        Returns:
            Health check status with diagnostics

        Example:
            >>> engine = ComplianceCheckerEngine()
            >>> health = engine.health_check()
            >>> assert health["status"] == "healthy"
        """
        try:
            # Check framework validators registered
            validators_count = len(self._framework_validators)
            expected_count = len(ComplianceFramework)

            # Check configuration
            config_valid = self.config is not None

            # Determine overall health
            is_healthy = validators_count == expected_count and config_valid

            return {
                "status": "healthy" if is_healthy else "degraded",
                "agent_id": AGENT_ID,
                "version": VERSION,
                "validators_registered": validators_count,
                "expected_validators": expected_count,
                "config_valid": config_valid,
                "frameworks_supported": [f.value for f in ComplianceFramework],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
