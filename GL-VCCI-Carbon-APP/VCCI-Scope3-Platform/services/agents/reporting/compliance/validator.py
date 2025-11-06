"""
Compliance Validator
GL-VCCI Scope 3 Platform

Validates data readiness and compliance for different reporting standards.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models import (
    ValidationCheck,
    ValidationResult,
    EmissionsData,
    EnergyData,
    CompanyInfo,
)
from ..config import (
    ReportStandard,
    ValidationLevel,
    QUALITY_THRESHOLDS,
    ESRS_E1_CONFIG,
    CDP_CONFIG,
    IFRS_S2_CONFIG,
    ISO_14083_CONFIG,
)
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class ComplianceValidator:
    """
    Validates data readiness and compliance for reporting standards.

    Features:
    - Standard-specific validation rules
    - Data quality checks
    - Completeness assessment
    - Compliance reporting
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize validator.

        Args:
            validation_level: Validation strictness level
        """
        self.validation_level = validation_level
        self.thresholds = QUALITY_THRESHOLDS

    def validate_for_esrs_e1(
        self,
        emissions_data: EmissionsData,
        energy_data: Optional[EnergyData] = None,
        company_info: Optional[CompanyInfo] = None,
    ) -> ValidationResult:
        """
        Validate data readiness for ESRS E1 (EU CSRD) reporting.

        Args:
            emissions_data: Emissions data
            energy_data: Energy consumption data
            company_info: Company information

        Returns:
            ValidationResult with compliance status
        """
        checks: List[ValidationCheck] = []

        logger.info("Validating data for ESRS E1 (EU CSRD) compliance")

        # Check 1: All scopes present
        checks.append(self._check_all_scopes_present(emissions_data))

        # Check 2: Scope 3 coverage
        checks.append(self._check_scope3_coverage(emissions_data))

        # Check 3: Data quality
        checks.append(self._check_data_quality(emissions_data))

        # Check 4: Intensity metrics
        checks.append(self._check_intensity_metrics(emissions_data, company_info))

        # Check 5: Energy data (E1-5)
        if energy_data:
            checks.append(self._check_energy_data(energy_data))
        else:
            checks.append(
                ValidationCheck(
                    check_name="E1-5: Energy Data",
                    status="WARNING",
                    message="Energy consumption data not provided",
                    severity="warning",
                )
            )

        # Check 6: Year-over-year comparison
        checks.append(self._check_yoy_data(emissions_data))

        # Check 7: Methodology documentation
        checks.append(self._check_methodology_documentation(emissions_data))

        # Calculate results
        return self._compile_validation_result(
            checks=checks,
            standard=ReportStandard.ESRS_E1,
            emissions_data=emissions_data,
        )

    def validate_for_cdp(
        self,
        emissions_data: EmissionsData,
        energy_data: Optional[EnergyData] = None,
    ) -> ValidationResult:
        """
        Validate data readiness for CDP reporting.

        Args:
            emissions_data: Emissions data
            energy_data: Energy consumption data

        Returns:
            ValidationResult with CDP readiness
        """
        checks: List[ValidationCheck] = []

        logger.info("Validating data for CDP compliance")

        # Check 1: All scopes present (C6.1-C6.5)
        checks.append(self._check_all_scopes_present(emissions_data))

        # Check 2: Required Scope 3 categories (Cat 1, 4, 6 minimum)
        checks.append(self._check_cdp_scope3_categories(emissions_data))

        # Check 3: Methodology per category
        checks.append(self._check_category_methodologies(emissions_data))

        # Check 4: Data quality indicators
        checks.append(self._check_data_quality(emissions_data))

        # Check 5: Energy consumption (C8)
        if energy_data:
            checks.append(self._check_energy_data(energy_data))
        else:
            checks.append(
                ValidationCheck(
                    check_name="C8: Energy Data",
                    status="WARNING",
                    message="Energy data not provided (CDP C8)",
                    severity="warning",
                )
            )

        # Check 6: Uncertainty quantification (C9)
        checks.append(self._check_uncertainty_data(emissions_data))

        # Calculate results
        return self._compile_validation_result(
            checks=checks,
            standard=ReportStandard.CDP,
            emissions_data=emissions_data,
        )

    def validate_for_ifrs_s2(
        self,
        emissions_data: EmissionsData,
        risks_opportunities: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate data readiness for IFRS S2 climate disclosures.

        Args:
            emissions_data: Emissions data
            risks_opportunities: Climate risks and opportunities

        Returns:
            ValidationResult with IFRS S2 compliance
        """
        checks: List[ValidationCheck] = []

        logger.info("Validating data for IFRS S2 compliance")

        # Check 1: Cross-industry metrics (Scope 1, 2, 3)
        checks.append(self._check_all_scopes_present(emissions_data))

        # Check 2: Data quality and reliability
        checks.append(self._check_data_quality(emissions_data))

        # Check 3: Climate risks and opportunities
        if risks_opportunities:
            checks.append(
                ValidationCheck(
                    check_name="Climate Risks & Opportunities",
                    status="PASS",
                    message="Climate risks and opportunities documented",
                    severity="info",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check_name="Climate Risks & Opportunities",
                    status="FAIL" if self.validation_level == ValidationLevel.STRICT else "WARNING",
                    message="Climate risks and opportunities not provided",
                    severity="error" if self.validation_level == ValidationLevel.STRICT else "warning",
                )
            )

        # Check 4: Financial impact assessment
        checks.append(self._check_financial_data(emissions_data))

        # Calculate results
        return self._compile_validation_result(
            checks=checks,
            standard=ReportStandard.IFRS_S2,
            emissions_data=emissions_data,
        )

    def validate_for_iso_14083(
        self,
        transport_data: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate data for ISO 14083 conformance certificate.

        Args:
            transport_data: Transport emissions data

        Returns:
            ValidationResult with ISO 14083 conformance
        """
        checks: List[ValidationCheck] = []

        logger.info("Validating data for ISO 14083 conformance")

        # Check 1: Transport mode breakdown
        if "transport_by_mode" in transport_data and transport_data["transport_by_mode"]:
            checks.append(
                ValidationCheck(
                    check_name="Transport Mode Breakdown",
                    status="PASS",
                    message=f"{len(transport_data['transport_by_mode'])} transport modes documented",
                    severity="info",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check_name="Transport Mode Breakdown",
                    status="FAIL",
                    message="Transport mode breakdown missing",
                    severity="error",
                )
            )

        # Check 2: Emission factors documentation
        if "emission_factors_used" in transport_data and transport_data["emission_factors_used"]:
            checks.append(
                ValidationCheck(
                    check_name="Emission Factors",
                    status="PASS",
                    message="Emission factors documented with sources",
                    severity="info",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check_name="Emission Factors",
                    status="FAIL",
                    message="Emission factor documentation missing",
                    severity="error",
                )
            )

        # Check 3: Calculation methodology
        methodology = transport_data.get("methodology", "")
        if "ISO 14083" in methodology:
            checks.append(
                ValidationCheck(
                    check_name="Methodology Conformance",
                    status="PASS",
                    message=f"Methodology: {methodology}",
                    severity="info",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check_name="Methodology Conformance",
                    status="WARNING",
                    message="ISO 14083 methodology not explicitly declared",
                    severity="warning",
                )
            )

        # Check 4: Data quality assessment
        dqi = transport_data.get("data_quality_score", 0)
        if dqi >= self.thresholds["min_dqi_score"]:
            checks.append(
                ValidationCheck(
                    check_name="Data Quality",
                    status="PASS",
                    message=f"Data quality score: {dqi:.1f}/100",
                    severity="info",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    check_name="Data Quality",
                    status="WARNING",
                    message=f"Data quality score {dqi:.1f} below threshold {self.thresholds['min_dqi_score']}",
                    severity="warning",
                )
            )

        # Compile result
        passed = sum(1 for c in checks if c.status == "PASS")
        failed = sum(1 for c in checks if c.status == "FAIL")
        warnings = sum(1 for c in checks if c.status == "WARNING")

        is_valid = failed == 0 if self.validation_level == ValidationLevel.STRICT else True

        return ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            checks=checks,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            standard=ReportStandard.ISO_14083,
            completeness_pct=100.0 * passed / len(checks) if checks else 0.0,
        )

    # ========================================================================
    # INTERNAL CHECK METHODS
    # ========================================================================

    def _check_all_scopes_present(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check that Scope 1, 2, 3 are all present."""
        if (
            emissions_data.scope1_tco2e >= 0
            and emissions_data.scope2_location_tco2e >= 0
            and emissions_data.scope3_tco2e >= 0
        ):
            return ValidationCheck(
                check_name="Scopes 1-3 Present",
                status="PASS",
                message="All scopes (1, 2, 3) have emissions data",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="Scopes 1-3 Present",
                status="FAIL",
                message="Missing one or more scope emissions data",
                severity="error",
            )

    def _check_scope3_coverage(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check Scope 3 category coverage."""
        categories_covered = len(emissions_data.scope3_categories)
        total_categories = 15
        coverage = categories_covered / total_categories

        if coverage >= self.thresholds["min_scope_coverage"]:
            return ValidationCheck(
                check_name="Scope 3 Coverage",
                status="PASS",
                message=f"{categories_covered}/15 categories covered ({coverage:.0%})",
                severity="info",
                details={"coverage": coverage, "categories": list(emissions_data.scope3_categories.keys())},
            )
        else:
            return ValidationCheck(
                check_name="Scope 3 Coverage",
                status="WARNING",
                message=f"Only {categories_covered}/15 categories covered ({coverage:.0%}), below {self.thresholds['min_scope_coverage']:.0%}",
                severity="warning",
                details={"coverage": coverage, "categories": list(emissions_data.scope3_categories.keys())},
            )

    def _check_data_quality(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check average data quality score."""
        dqi = emissions_data.avg_dqi_score

        if dqi >= self.thresholds["min_dqi_score"]:
            rating = "Excellent" if dqi >= 90 else "Good"
            return ValidationCheck(
                check_name="Data Quality",
                status="PASS",
                message=f"Average DQI: {dqi:.1f}/100 ({rating})",
                severity="info",
                details={"dqi_score": dqi},
            )
        else:
            return ValidationCheck(
                check_name="Data Quality",
                status="WARNING",
                message=f"Average DQI {dqi:.1f} below threshold {self.thresholds['min_dqi_score']}",
                severity="warning",
                details={"dqi_score": dqi, "threshold": self.thresholds["min_dqi_score"]},
            )

    def _check_intensity_metrics(
        self, emissions_data: EmissionsData, company_info: Optional[CompanyInfo]
    ) -> ValidationCheck:
        """Check if intensity metrics can be calculated."""
        if not company_info:
            return ValidationCheck(
                check_name="Intensity Metrics",
                status="WARNING",
                message="Company info not provided, cannot calculate intensity metrics",
                severity="warning",
            )

        has_revenue = company_info.annual_revenue_usd is not None and company_info.annual_revenue_usd > 0
        has_employees = company_info.number_of_employees is not None and company_info.number_of_employees > 0

        if has_revenue and has_employees:
            return ValidationCheck(
                check_name="Intensity Metrics",
                status="PASS",
                message="Revenue and employee data available for intensity calculations",
                severity="info",
            )
        else:
            missing = []
            if not has_revenue:
                missing.append("revenue")
            if not has_employees:
                missing.append("employees")

            return ValidationCheck(
                check_name="Intensity Metrics",
                status="WARNING",
                message=f"Missing data for intensity metrics: {', '.join(missing)}",
                severity="warning",
            )

    def _check_energy_data(self, energy_data: EnergyData) -> ValidationCheck:
        """Check energy consumption data."""
        if energy_data.total_energy_mwh > 0:
            return ValidationCheck(
                check_name="Energy Consumption",
                status="PASS",
                message=f"Total energy: {energy_data.total_energy_mwh:,.0f} MWh, Renewable: {energy_data.renewable_pct or 0:.1f}%",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="Energy Consumption",
                status="WARNING",
                message="Energy consumption data not provided or zero",
                severity="warning",
            )

    def _check_yoy_data(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check year-over-year comparison data."""
        if emissions_data.prior_year_emissions and emissions_data.yoy_change_pct is not None:
            return ValidationCheck(
                check_name="Year-over-Year Comparison",
                status="PASS",
                message=f"Prior year data available, YoY change: {emissions_data.yoy_change_pct:+.1f}%",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="Year-over-Year Comparison",
                status="WARNING",
                message="Prior year emissions data not available",
                severity="warning",
            )

    def _check_methodology_documentation(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check methodology documentation."""
        if emissions_data.provenance_chains:
            return ValidationCheck(
                check_name="Methodology Documentation",
                status="PASS",
                message=f"{len(emissions_data.provenance_chains)} provenance chains documented",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="Methodology Documentation",
                status="WARNING",
                message="Provenance chains not provided",
                severity="warning",
            )

    def _check_cdp_scope3_categories(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check CDP required Scope 3 categories (1, 4, 6)."""
        required_cats = set(CDP_CONFIG["required_categories"])
        available_cats = set(emissions_data.scope3_categories.keys())
        missing_cats = required_cats - available_cats

        if not missing_cats:
            return ValidationCheck(
                check_name="CDP Required Categories",
                status="PASS",
                message="All required Scope 3 categories present (Cat 1, 4, 6)",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="CDP Required Categories",
                status="FAIL",
                message=f"Missing required categories: {sorted(missing_cats)}",
                severity="error",
                details={"missing": sorted(missing_cats)},
            )

    def _check_category_methodologies(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check that methodologies are documented per category."""
        if emissions_data.scope3_details:
            return ValidationCheck(
                check_name="Category Methodologies",
                status="PASS",
                message="Detailed methodology available per category",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="Category Methodologies",
                status="WARNING",
                message="Detailed methodology per category not provided",
                severity="warning",
            )

    def _check_uncertainty_data(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check uncertainty quantification."""
        if emissions_data.uncertainty_results:
            return ValidationCheck(
                check_name="Uncertainty Quantification",
                status="PASS",
                message="Uncertainty analysis available",
                severity="info",
            )
        else:
            return ValidationCheck(
                check_name="Uncertainty Quantification",
                status="WARNING",
                message="Uncertainty analysis not provided",
                severity="warning",
            )

    def _check_financial_data(self, emissions_data: EmissionsData) -> ValidationCheck:
        """Check financial impact data."""
        # This would check for financial impact assessments
        return ValidationCheck(
            check_name="Financial Impact",
            status="WARNING",
            message="Financial impact assessment requires additional qualitative data",
            severity="warning",
        )

    def _compile_validation_result(
        self,
        checks: List[ValidationCheck],
        standard: ReportStandard,
        emissions_data: EmissionsData,
    ) -> ValidationResult:
        """Compile final validation result."""
        passed = sum(1 for c in checks if c.status == "PASS")
        failed = sum(1 for c in checks if c.status == "FAIL")
        warnings = sum(1 for c in checks if c.status == "WARNING")

        # Determine validity based on level
        if self.validation_level == ValidationLevel.STRICT:
            is_valid = failed == 0 and warnings == 0
        elif self.validation_level == ValidationLevel.STANDARD:
            is_valid = failed == 0
        else:  # LENIENT
            is_valid = True

        completeness = 100.0 * passed / len(checks) if checks else 0.0

        # Generate recommendations
        recommendations = []
        for check in checks:
            if check.status in ["FAIL", "WARNING"]:
                recommendations.append(f"{check.check_name}: {check.message}")

        return ValidationResult(
            is_valid=is_valid,
            validation_level=self.validation_level,
            checks=checks,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            standard=standard,
            completeness_pct=completeness,
            recommendations=recommendations[:10],  # Top 10
        )


__all__ = ["ComplianceValidator"]
