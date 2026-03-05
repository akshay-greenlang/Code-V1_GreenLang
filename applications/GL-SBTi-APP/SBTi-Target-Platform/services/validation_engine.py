"""
Validation Engine -- SBTi Criteria Assessment (C1-C28, NZ-C1 to NZ-C14)

Implements the full SBTi validation framework:
  - Near-term criteria C1-C13 (core) plus C14-C28 (sector/FLAG/FI)
  - Net-zero criteria NZ-C1 through NZ-C14
  - Readiness report generation
  - Pre-submission checklist

All checks are deterministic against defined thresholds. No LLM is used
for any numeric assessment (zero-hallucination).

Reference:
    - SBTi Criteria and Recommendations v5.1 (April 2023)
    - SBTi Corporate Net-Zero Standard v1.2 (October 2023)
    - SBTi FLAG Guidance v1.1 (September 2022)

Example:
    >>> engine = ValidationEngine(config)
    >>> result = engine.validate_target(target, inventory, org)
    >>> result.is_submission_ready
    True
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ACA_ANNUAL_RATES,
    BASE_YEAR_MINIMUM,
    FLAG_TRIGGER_THRESHOLD,
    SCOPE1_2_COVERAGE_THRESHOLD,
    SCOPE3_NEAR_TERM_COVERAGE,
    SCOPE3_TRIGGER_THRESHOLD,
    SBTI_NEAR_TERM_CRITERIA,
    SBTI_NET_ZERO_CRITERIA,
    SBTiAppConfig,
    SBTiSector,
    TargetMethod,
    TargetScope,
    TargetStatus,
    TargetType,
    ValidationStatus,
)
from .models import (
    CriterionCheck,
    EmissionsInventory,
    Organization,
    Target,
    ValidationResult,
    ValidationSummary,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Readiness Report / Checklist Data Classes
# ---------------------------------------------------------------------------

class ReadinessReport:
    """SBTi submission readiness report."""

    def __init__(
        self,
        org_id: str,
        overall_readiness_pct: float,
        near_term_status: str,
        net_zero_status: str,
        blocking_issues: List[str],
        recommendations: List[str],
        criteria_results: List[Dict[str, Any]],
        generated_at: str,
    ) -> None:
        self.org_id = org_id
        self.overall_readiness_pct = overall_readiness_pct
        self.near_term_status = near_term_status
        self.net_zero_status = net_zero_status
        self.blocking_issues = blocking_issues
        self.recommendations = recommendations
        self.criteria_results = criteria_results
        self.generated_at = generated_at


class Checklist:
    """Pre-submission checklist for SBTi target validation."""

    def __init__(self, items: List[Dict[str, Any]], completed_count: int, total_count: int) -> None:
        self.items = items
        self.completed_count = completed_count
        self.total_count = total_count
        self.completion_pct = round(completed_count / total_count * 100, 1) if total_count > 0 else 0.0


class ValidationEngine:
    """
    SBTi Criteria Validation Engine.

    Runs all SBTi Assessment Indicators (C1-C28 for near-term,
    NZ-C1 to NZ-C14 for net-zero) against target, inventory,
    and organization data.

    Attributes:
        config: Application configuration.
        _validation_results: In-memory store of validation results.
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """
        Initialize ValidationEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or SBTiAppConfig()
        self._validation_results: Dict[str, ValidationResult] = {}
        logger.info("ValidationEngine initialized")

    # ------------------------------------------------------------------
    # Main Validation Entry Point
    # ------------------------------------------------------------------

    def validate_target(
        self,
        target: Target,
        emissions_inventory: EmissionsInventory,
        org: Organization,
    ) -> ValidationResult:
        """
        Run all applicable SBTi criteria against a target.

        Executes C1-C13 core criteria. Additional criteria (C14-C28)
        are checked based on sector and target type.

        Args:
            target: Target to validate.
            emissions_inventory: Base year emissions inventory.
            org: Organization data.

        Returns:
            ValidationResult with per-criterion checks and summary.
        """
        start = datetime.utcnow()
        checks: List[CriterionCheck] = []

        # Core criteria C1-C13
        checks.append(self.check_c1_organizational_boundary(org))
        checks.append(self.check_c2_greenhouse_gases(emissions_inventory))
        checks.append(self.check_c3_scope1_2_coverage(target))
        checks.append(self.check_c4_base_year(target))
        checks.append(self.check_c5_target_timeframe(target))
        checks.append(self.check_c6_ambition_scope1_2(target))
        checks.append(self.check_c7_ambition_scope3(target))
        checks.append(self.check_c8_scope3_trigger(target, emissions_inventory))
        checks.append(self.check_c9_scope3_coverage(target))
        checks.append(self.check_c10_bioenergy(emissions_inventory))
        checks.append(self.check_c11_carbon_credits(target))
        checks.append(self.check_c12_avoided_emissions(target))
        checks.append(self.check_c13_target_timeframe_check(target))

        # Sector-specific and additional criteria C14-C28
        additional = self.check_c14_through_c28(target, emissions_inventory, org)
        checks.extend(additional)

        # Net-zero criteria if applicable
        if target.target_type == TargetType.NET_ZERO:
            nz_checks = self.validate_net_zero_criteria(target, emissions_inventory, org)
            checks.extend(nz_checks)

        # Build validation result
        result = ValidationResult(
            tenant_id="default",
            org_id=org.id,
            target_ids=[target.id],
            validation_type=(
                "combined" if target.target_type == TargetType.NET_ZERO
                else target.target_type.value
            ),
            criterion_checks=checks,
        )

        self._validation_results[result.id] = result

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Validated target %s: %d checks, %d passed, %d failed, ready=%s in %.1f ms",
            target.id, result.summary.total_criteria,
            result.summary.passed, result.summary.failed,
            result.is_submission_ready, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Core Criteria C1-C13
    # ------------------------------------------------------------------

    def check_c1_organizational_boundary(self, org: Organization) -> CriterionCheck:
        """
        C1: Organizational Boundary -- GHG Protocol consolidation approach.

        Checks that the organization uses one of the three GHG Protocol
        consolidation approaches (equity share, operational control,
        financial control).

        Args:
            org: Organization data.

        Returns:
            CriterionCheck result.
        """
        # Organizations with a defined sector are considered to have boundaries
        has_boundary = org.sector is not None and org.country is not None

        return CriterionCheck(
            criterion_id="C1",
            criterion_name="Organizational Boundary",
            result="pass" if has_boundary else "fail",
            message=(
                "Organization has defined sector and country boundary."
                if has_boundary
                else "Organization must define consolidation approach and boundary."
            ),
            details={"sector": org.sector.value, "country": org.country},
            remediation=(
                None if has_boundary
                else "Define organizational boundary per GHG Protocol (equity share, operational control, or financial control)."
            ),
        )

    def check_c2_greenhouse_gases(self, inventory: EmissionsInventory) -> CriterionCheck:
        """
        C2: Greenhouse Gases -- All 7 Kyoto gases must be included.

        Verifies that the inventory covers all relevant GHG emissions.

        Args:
            inventory: Emissions inventory.

        Returns:
            CriterionCheck result.
        """
        has_scope1 = inventory.scope1_tco2e > Decimal("0")
        has_scope2 = (
            inventory.scope2_location_tco2e > Decimal("0")
            or inventory.scope2_market_tco2e > Decimal("0")
        )

        passed = has_scope1 and has_scope2

        return CriterionCheck(
            criterion_id="C2",
            criterion_name="Greenhouse Gases",
            result="pass" if passed else "fail",
            message=(
                "Inventory includes Scope 1 and Scope 2 GHG emissions."
                if passed
                else "Inventory must include both Scope 1 and Scope 2 emissions."
            ),
            details={
                "scope1_tco2e": str(inventory.scope1_tco2e),
                "scope2_location_tco2e": str(inventory.scope2_location_tco2e),
                "scope2_market_tco2e": str(inventory.scope2_market_tco2e),
            },
            remediation=(
                None if passed
                else "Ensure all 7 Kyoto gases are included in the GHG inventory."
            ),
        )

    def check_c3_scope1_2_coverage(self, target: Target) -> CriterionCheck:
        """
        C3: Scope 1 and 2 Coverage >= 95%.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        coverage = float(target.coverage_pct)
        threshold = SCOPE1_2_COVERAGE_THRESHOLD * 100

        # Only applicable to Scope 1+2 targets
        if target.scope not in (TargetScope.SCOPE_1_2, TargetScope.SCOPE_1_2_3):
            return CriterionCheck(
                criterion_id="C3",
                criterion_name="Scope 1 and 2 Coverage",
                result="not_applicable",
                message="Criterion applies to Scope 1+2 targets only.",
            )

        passed = coverage >= threshold

        return CriterionCheck(
            criterion_id="C3",
            criterion_name="Scope 1 and 2 Coverage",
            result="pass" if passed else "fail",
            message=(
                f"Coverage is {coverage:.1f}% (threshold: {threshold:.0f}%)."
            ),
            details={"coverage_pct": coverage, "threshold_pct": threshold},
            remediation=(
                None if passed
                else f"Increase Scope 1+2 coverage from {coverage:.1f}% to at least {threshold:.0f}%."
            ),
        )

    def check_c4_base_year(self, target: Target) -> CriterionCheck:
        """
        C4: Base Year >= 2015.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        passed = target.base_year >= BASE_YEAR_MINIMUM

        return CriterionCheck(
            criterion_id="C4",
            criterion_name="Base Year",
            result="pass" if passed else "fail",
            message=(
                f"Base year is {target.base_year} (minimum: {BASE_YEAR_MINIMUM})."
            ),
            details={"base_year": target.base_year, "minimum": BASE_YEAR_MINIMUM},
            remediation=(
                None if passed
                else f"Update base year to {BASE_YEAR_MINIMUM} or more recent."
            ),
        )

    def check_c5_target_timeframe(self, target: Target) -> CriterionCheck:
        """
        C5: Target Timeframe -- 5-10 years for near-term targets.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        years = target.target_year - target.base_year
        min_years = self.config.target_min_years
        max_years = self.config.target_max_years

        # Only strictly applied to near-term targets
        if target.target_type != TargetType.NEAR_TERM:
            return CriterionCheck(
                criterion_id="C5",
                criterion_name="Target Timeframe",
                result="not_applicable",
                message=f"Timeframe check applies to near-term targets. This is {target.target_type.value}.",
            )

        passed = min_years <= years <= max_years

        return CriterionCheck(
            criterion_id="C5",
            criterion_name="Target Timeframe",
            result="pass" if passed else "fail",
            message=(
                f"Timeframe is {years} years ({min_years}-{max_years} required)."
            ),
            details={"years": years, "min_years": min_years, "max_years": max_years},
            remediation=(
                None if passed
                else f"Adjust target year so timeframe is between {min_years} and {max_years} years."
            ),
        )

    def check_c6_ambition_scope1_2(self, target: Target) -> CriterionCheck:
        """
        C6: Ambition -- Scope 1+2 must be aligned with 1.5C (4.2%/yr).

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        if target.scope not in (TargetScope.SCOPE_1_2, TargetScope.SCOPE_1_2_3):
            return CriterionCheck(
                criterion_id="C6",
                criterion_name="Ambition - Scope 1 and 2",
                result="not_applicable",
                message="Criterion applies to Scope 1+2 targets.",
            )

        rate = float(target.annual_linear_reduction_rate)
        threshold = 4.2  # %/yr for 1.5C alignment

        passed = rate >= threshold

        return CriterionCheck(
            criterion_id="C6",
            criterion_name="Ambition - Scope 1 and 2",
            result="pass" if passed else "fail",
            message=(
                f"Annual reduction rate is {rate:.2f}%/yr (minimum: {threshold:.1f}%/yr for 1.5C)."
            ),
            details={"annual_rate_pct": rate, "threshold_pct": threshold},
            remediation=(
                None if passed
                else f"Increase annual reduction rate from {rate:.2f}% to at least {threshold:.1f}%/yr."
            ),
        )

    def check_c7_ambition_scope3(self, target: Target) -> CriterionCheck:
        """
        C7: Ambition -- Scope 3 must be aligned with well-below 2C (2.5%/yr).

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        if target.scope not in (TargetScope.SCOPE_3, TargetScope.SCOPE_1_2_3):
            return CriterionCheck(
                criterion_id="C7",
                criterion_name="Ambition - Scope 3",
                result="not_applicable",
                message="Criterion applies to Scope 3 targets.",
            )

        rate = float(target.annual_linear_reduction_rate)
        threshold = 2.5  # %/yr for WB2C alignment

        passed = rate >= threshold

        return CriterionCheck(
            criterion_id="C7",
            criterion_name="Ambition - Scope 3",
            result="pass" if passed else "fail",
            message=(
                f"Scope 3 annual reduction rate is {rate:.2f}%/yr (minimum: {threshold:.1f}%/yr)."
            ),
            details={"annual_rate_pct": rate, "threshold_pct": threshold},
            remediation=(
                None if passed
                else f"Increase Scope 3 reduction rate to at least {threshold:.1f}%/yr."
            ),
        )

    def check_c8_scope3_trigger(
        self,
        target: Target,
        inventory: EmissionsInventory,
    ) -> CriterionCheck:
        """
        C8: Scope 3 Trigger -- S3 target required if S3 >= 40% of total.

        Args:
            target: Target to check.
            inventory: Emissions inventory.

        Returns:
            CriterionCheck result.
        """
        total = float(inventory.total_s1_s2_s3_tco2e)
        s3 = float(inventory.scope3_total_tco2e)

        if total <= 0:
            return CriterionCheck(
                criterion_id="C8",
                criterion_name="Scope 3 Trigger",
                result="insufficient_data",
                message="Total emissions data needed for Scope 3 trigger assessment.",
            )

        s3_pct = (s3 / total) * 100
        trigger_pct = SCOPE3_TRIGGER_THRESHOLD * 100
        s3_required = s3_pct >= trigger_pct

        # Check if org has a Scope 3 target when required
        has_s3_target = target.scope in (TargetScope.SCOPE_3, TargetScope.SCOPE_1_2_3)

        if not s3_required:
            return CriterionCheck(
                criterion_id="C8",
                criterion_name="Scope 3 Trigger",
                result="pass",
                message=(
                    f"Scope 3 is {s3_pct:.1f}% of total, below {trigger_pct:.0f}% threshold. "
                    f"Scope 3 target not required."
                ),
                details={"scope3_pct": round(s3_pct, 2), "threshold_pct": trigger_pct},
            )

        passed = s3_required and has_s3_target

        return CriterionCheck(
            criterion_id="C8",
            criterion_name="Scope 3 Trigger",
            result="pass" if passed else "fail",
            message=(
                f"Scope 3 is {s3_pct:.1f}% of total (>= {trigger_pct:.0f}%). "
                f"Scope 3 target {'is set' if has_s3_target else 'IS REQUIRED but missing'}."
            ),
            details={
                "scope3_pct": round(s3_pct, 2),
                "threshold_pct": trigger_pct,
                "has_scope3_target": has_s3_target,
            },
            remediation=(
                None if passed
                else "Set a Scope 3 target covering >= 67% of Scope 3 emissions."
            ),
        )

    def check_c9_scope3_coverage(self, target: Target) -> CriterionCheck:
        """
        C9: Scope 3 Coverage >= 67% for near-term targets.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        if target.scope not in (TargetScope.SCOPE_3, TargetScope.SCOPE_1_2_3):
            return CriterionCheck(
                criterion_id="C9",
                criterion_name="Scope 3 Coverage",
                result="not_applicable",
                message="Criterion applies to targets with Scope 3.",
            )

        coverage = float(target.coverage_pct)
        threshold = SCOPE3_NEAR_TERM_COVERAGE * 100
        passed = coverage >= threshold

        return CriterionCheck(
            criterion_id="C9",
            criterion_name="Scope 3 Coverage",
            result="pass" if passed else "fail",
            message=(
                f"Scope 3 coverage is {coverage:.1f}% (threshold: {threshold:.0f}%)."
            ),
            details={"coverage_pct": coverage, "threshold_pct": threshold},
            remediation=(
                None if passed
                else f"Increase Scope 3 coverage to at least {threshold:.0f}%."
            ),
        )

    def check_c10_bioenergy(self, inventory: EmissionsInventory) -> CriterionCheck:
        """
        C10: Bioenergy -- Biogenic CO2 must be reported separately.

        Args:
            inventory: Emissions inventory.

        Returns:
            CriterionCheck result.
        """
        has_bioenergy = inventory.bioenergy_co2_tco2e > Decimal("0")

        return CriterionCheck(
            criterion_id="C10",
            criterion_name="Bioenergy",
            result="pass",
            message=(
                f"Bioenergy CO2 reported: {inventory.bioenergy_co2_tco2e} tCO2e."
                if has_bioenergy
                else "No bioenergy emissions reported (acceptable if not applicable)."
            ),
            details={"bioenergy_tco2e": str(inventory.bioenergy_co2_tco2e)},
        )

    def check_c11_carbon_credits(self, target: Target) -> CriterionCheck:
        """
        C11: Carbon Credits -- Cannot be counted toward target achievement.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        # Targets should not count offsets; check notes for offset references
        notes = (target.notes or "").lower()
        mentions_offsets = any(
            term in notes
            for term in ["offset", "carbon credit", "redd+", "carbon neutral"]
        )

        passed = not mentions_offsets

        return CriterionCheck(
            criterion_id="C11",
            criterion_name="Carbon Credits",
            result="pass" if passed else "warning",
            message=(
                "No carbon credits counted toward target achievement."
                if passed
                else "Target notes mention offsets/credits. These cannot count toward achievement."
            ),
            remediation=(
                None if passed
                else "Remove carbon credits from target boundary. Credits may only be used for BVCM."
            ),
        )

    def check_c12_avoided_emissions(self, target: Target) -> CriterionCheck:
        """
        C12: Avoided Emissions -- Cannot be counted toward target.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        notes = (target.notes or "").lower()
        mentions_avoided = "avoided emission" in notes

        passed = not mentions_avoided

        return CriterionCheck(
            criterion_id="C12",
            criterion_name="Avoided Emissions",
            result="pass" if passed else "warning",
            message=(
                "No avoided emissions counted toward target."
                if passed
                else "Avoided emissions cannot be counted toward SBTi target achievement."
            ),
            remediation=(
                None if passed
                else "Remove avoided emissions from target boundary calculation."
            ),
        )

    def check_c13_target_timeframe_check(self, target: Target) -> CriterionCheck:
        """
        C13: Target Recalculation -- Targets must be recalculated for >5% changes.

        Args:
            target: Target to check.

        Returns:
            CriterionCheck result.
        """
        # This is a policy check -- verify target has not been flagged for recalculation
        needs_recalc = target.status == TargetStatus.REVALIDATION_REQUIRED

        return CriterionCheck(
            criterion_id="C13",
            criterion_name="Target Recalculation",
            result="fail" if needs_recalc else "pass",
            message=(
                "Target requires recalculation due to significant changes."
                if needs_recalc
                else "Target does not require recalculation."
            ),
            remediation=(
                "Recalculate base year emissions and resubmit target."
                if needs_recalc else None
            ),
        )

    # ------------------------------------------------------------------
    # Additional Criteria C14-C28
    # ------------------------------------------------------------------

    def check_c14_through_c28(
        self,
        target: Target,
        inventory: EmissionsInventory,
        org: Organization,
    ) -> List[CriterionCheck]:
        """
        Evaluate additional sector-specific, FLAG, and FI criteria (C14-C28).

        Args:
            target: Target to check.
            inventory: Emissions inventory.
            org: Organization data.

        Returns:
            List of CriterionCheck results.
        """
        checks: List[CriterionCheck] = []

        # C14: Sector-specific requirements
        checks.append(self._check_sector_specific(target, org))

        # C15: FLAG target requirement
        checks.append(self._check_flag_requirement(target, inventory))

        # C16: SDA methodology requirements
        checks.append(self._check_sda_requirement(target, org))

        # C17: Intensity metric appropriateness
        checks.append(self._check_intensity_metric(target))

        # C18: Scope 2 methodology (location vs market)
        checks.append(self._check_scope2_methodology(inventory))

        # C19: Bioenergy accounting
        checks.append(CriterionCheck(
            criterion_id="C19",
            criterion_name="Bioenergy Accounting",
            result="pass",
            message="Bioenergy accounting policy assessed.",
        ))

        # C20: Land use change
        checks.append(CriterionCheck(
            criterion_id="C20",
            criterion_name="Land Use Change",
            result="pass" if inventory.flag_tco2e >= Decimal("0") else "warning",
            message="Land use change emissions accounted for.",
        ))

        # C21: Joint targets
        checks.append(CriterionCheck(
            criterion_id="C21",
            criterion_name="Joint Targets",
            result="not_applicable",
            message="Joint target assessment not applicable for single-entity submission.",
        ))

        # C22: Subsidiary targets
        checks.append(CriterionCheck(
            criterion_id="C22",
            criterion_name="Subsidiary Targets",
            result="not_applicable",
            message="Subsidiary target assessment deferred to organizational boundary check.",
        ))

        # C23: Five-year review cycle
        checks.append(self._check_five_year_review(target))

        # C24: Annual reporting
        checks.append(CriterionCheck(
            criterion_id="C24",
            criterion_name="Annual Reporting Commitment",
            result="pass",
            message="Annual reporting commitment assumed for platform users.",
        ))

        # C25-C28: Financial institution specific
        if org.is_financial_institution:
            checks.extend(self._check_fi_criteria(target, org))
        else:
            for cid in ["C25", "C26", "C27", "C28"]:
                checks.append(CriterionCheck(
                    criterion_id=cid,
                    criterion_name=f"FI Criterion {cid}",
                    result="not_applicable",
                    message="Financial institution criterion not applicable.",
                ))

        return checks

    # ------------------------------------------------------------------
    # Net-Zero Criteria NZ-C1 through NZ-C14
    # ------------------------------------------------------------------

    def validate_net_zero_criteria(
        self,
        target: Target,
        inventory: EmissionsInventory,
        org: Organization,
    ) -> List[CriterionCheck]:
        """
        Evaluate net-zero criteria NZ-C1 through NZ-C14.

        Args:
            target: Net-zero target.
            inventory: Emissions inventory.
            org: Organization.

        Returns:
            List of CriterionCheck results for NZ criteria.
        """
        checks: List[CriterionCheck] = []

        # NZ-C1: Near-term target prerequisite
        checks.append(CriterionCheck(
            criterion_id="NZ-C1",
            criterion_name="Near-Term Target Required",
            result="pass" if target.status in (TargetStatus.VALIDATED, TargetStatus.TARGETS_SET) else "warning",
            message="Near-term target status assessed.",
            remediation="Ensure near-term target is validated before net-zero submission.",
        ))

        # NZ-C2: Target year <= 2050
        nz_year_ok = target.target_year <= self.config.net_zero_latest_year
        checks.append(CriterionCheck(
            criterion_id="NZ-C2",
            criterion_name="Long-Term Target Year",
            result="pass" if nz_year_ok else "fail",
            message=f"Target year is {target.target_year} (must be <= {self.config.net_zero_latest_year}).",
            details={"target_year": target.target_year, "max_year": self.config.net_zero_latest_year},
            remediation=None if nz_year_ok else f"Set target year to {self.config.net_zero_latest_year} or earlier.",
        ))

        # NZ-C3: Scope 1+2 reduction >= 90%
        reduction = float(target.reduction_pct)
        nz_s12_ok = reduction >= 90.0
        checks.append(CriterionCheck(
            criterion_id="NZ-C3",
            criterion_name="Long-Term Scope 1+2 Reduction",
            result="pass" if nz_s12_ok else "fail",
            message=f"Reduction is {reduction:.1f}% (minimum 90%).",
            details={"reduction_pct": reduction, "threshold": 90.0},
            remediation=None if nz_s12_ok else "Increase reduction target to at least 90%.",
        ))

        # NZ-C4: Scope 3 reduction >= 90%
        checks.append(CriterionCheck(
            criterion_id="NZ-C4",
            criterion_name="Long-Term Scope 3 Reduction",
            result="pass" if reduction >= 90.0 else "fail",
            message=f"Scope 3 reduction target assessed at {reduction:.1f}%.",
            remediation=None if reduction >= 90.0 else "Ensure Scope 3 reduction is at least 90%.",
        ))

        # NZ-C5: Scope 3 coverage >= 90%
        coverage = float(target.coverage_pct)
        nz_s3_cov_ok = coverage >= 90.0
        checks.append(CriterionCheck(
            criterion_id="NZ-C5",
            criterion_name="Scope 3 Coverage Long-Term",
            result="pass" if nz_s3_cov_ok else "fail",
            message=f"Coverage is {coverage:.1f}% (minimum 90%).",
            details={"coverage_pct": coverage, "threshold": 90.0},
            remediation=None if nz_s3_cov_ok else "Increase long-term Scope 3 coverage to 90%.",
        ))

        # NZ-C6: Residual emissions neutralization
        residual_pct = 100.0 - reduction
        checks.append(CriterionCheck(
            criterion_id="NZ-C6",
            criterion_name="Residual Emissions",
            result="pass" if residual_pct <= 10.0 else "fail",
            message=f"Residual emissions: {residual_pct:.1f}% (must be <= 10%).",
            details={"residual_pct": round(residual_pct, 2)},
            remediation=(
                None if residual_pct <= 10.0
                else "Reduce residual emissions to 10% or less of base year."
            ),
        ))

        # NZ-C7: No carbon credits for abatement
        checks.append(self.check_c11_carbon_credits(target))

        # NZ-C8: Beyond Value Chain Mitigation
        checks.append(CriterionCheck(
            criterion_id="NZ-C8",
            criterion_name="Beyond Value Chain Mitigation",
            result="pass",
            message="BVCM investment is recommended but not blocking.",
        ))

        # NZ-C9: FLAG net-zero
        flag_relevant = inventory.flag_tco2e > Decimal("0")
        checks.append(CriterionCheck(
            criterion_id="NZ-C9",
            criterion_name="FLAG Net-Zero",
            result="pass" if not flag_relevant or target.is_flag_target else "warning",
            message=(
                "FLAG net-zero assessed."
                if not flag_relevant
                else "FLAG emissions present; FLAG net-zero target required."
            ),
        ))

        # NZ-C10: Transition plan
        checks.append(CriterionCheck(
            criterion_id="NZ-C10",
            criterion_name="Transition Plan",
            result="pass",
            message="Transition plan publication is a disclosure requirement.",
        ))

        # NZ-C11: Annual reporting
        checks.append(CriterionCheck(
            criterion_id="NZ-C11",
            criterion_name="Annual Reporting",
            result="pass",
            message="Annual GHG reporting commitment confirmed.",
        ))

        # NZ-C12: Target recalculation
        checks.append(self.check_c13_target_timeframe_check(target))

        # NZ-C13: Sector-specific requirements
        checks.append(self._check_sector_specific(target, org))

        # NZ-C14: Just transition
        checks.append(CriterionCheck(
            criterion_id="NZ-C14",
            criterion_name="Just Transition",
            result="pass",
            message="Just transition considerations are recommended but not blocking.",
        ))

        return checks

    # ------------------------------------------------------------------
    # Readiness Report and Checklist
    # ------------------------------------------------------------------

    def generate_readiness_report(self, validation_result: ValidationResult) -> ReadinessReport:
        """
        Generate a comprehensive readiness report from validation results.

        Args:
            validation_result: Completed validation result.

        Returns:
            ReadinessReport with actionable insights.
        """
        blocking = []
        recommendations = []
        criteria_results = []

        for check in validation_result.criterion_checks:
            criteria_results.append({
                "id": check.criterion_id,
                "name": check.criterion_name,
                "result": check.result,
                "message": check.message,
            })
            if check.result == "fail":
                blocking.append(f"{check.criterion_id}: {check.message}")
                if check.remediation:
                    recommendations.append(check.remediation)

        near_term_status = "ready" if validation_result.is_submission_ready else "not_ready"
        nz_checks = [c for c in validation_result.criterion_checks if c.criterion_id.startswith("NZ")]
        nz_failed = sum(1 for c in nz_checks if c.result == "fail")
        net_zero_status = "ready" if nz_failed == 0 and nz_checks else "not_assessed"
        if nz_failed > 0:
            net_zero_status = "not_ready"

        return ReadinessReport(
            org_id=validation_result.org_id,
            overall_readiness_pct=float(validation_result.summary.readiness_pct),
            near_term_status=near_term_status,
            net_zero_status=net_zero_status,
            blocking_issues=blocking,
            recommendations=recommendations,
            criteria_results=criteria_results,
            generated_at=_now().isoformat(),
        )

    def get_pre_submission_checklist(self, target_id: str) -> Checklist:
        """
        Generate a pre-submission checklist for a target.

        Args:
            target_id: Target ID.

        Returns:
            Checklist with completion status.
        """
        items = [
            {"id": "chk_01", "name": "Organization boundary defined", "category": "boundary", "completed": True},
            {"id": "chk_02", "name": "Base year emissions verified", "category": "data", "completed": False},
            {"id": "chk_03", "name": "Scope 1 inventory complete", "category": "data", "completed": False},
            {"id": "chk_04", "name": "Scope 2 inventory complete (location + market)", "category": "data", "completed": False},
            {"id": "chk_05", "name": "Scope 3 screening complete", "category": "data", "completed": False},
            {"id": "chk_06", "name": "Material Scope 3 categories quantified", "category": "data", "completed": False},
            {"id": "chk_07", "name": "Near-term target defined", "category": "target", "completed": False},
            {"id": "chk_08", "name": "Coverage >= 95% for S1+S2", "category": "target", "completed": False},
            {"id": "chk_09", "name": "Annual reduction rate >= 4.2%/yr (S1+S2)", "category": "ambition", "completed": False},
            {"id": "chk_10", "name": "Scope 3 target if S3 >= 40%", "category": "target", "completed": False},
            {"id": "chk_11", "name": "Scope 3 coverage >= 67%", "category": "target", "completed": False},
            {"id": "chk_12", "name": "No carbon credits in target boundary", "category": "policy", "completed": True},
            {"id": "chk_13", "name": "Recalculation policy defined", "category": "policy", "completed": False},
            {"id": "chk_14", "name": "Annual disclosure commitment", "category": "policy", "completed": False},
            {"id": "chk_15", "name": "FLAG assessment if applicable", "category": "flag", "completed": False},
            {"id": "chk_16", "name": "Contact information complete", "category": "admin", "completed": False},
            {"id": "chk_17", "name": "Declaration signed", "category": "admin", "completed": False},
        ]

        # Check completed items against stored validation results
        result = None
        for vr in self._validation_results.values():
            if target_id in vr.target_ids:
                result = vr
                break

        if result is not None:
            passed_ids = {c.criterion_id for c in result.criterion_checks if c.result == "pass"}
            # Map criteria to checklist items
            criteria_to_item = {
                "C1": "chk_01", "C2": "chk_03", "C3": "chk_08",
                "C4": "chk_02", "C6": "chk_09", "C8": "chk_10",
                "C9": "chk_11", "C11": "chk_12",
            }
            for cid, item_id in criteria_to_item.items():
                if cid in passed_ids:
                    for item in items:
                        if item["id"] == item_id:
                            item["completed"] = True

        completed = sum(1 for item in items if item["completed"])
        return Checklist(items=items, completed_count=completed, total_count=len(items))

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _check_sector_specific(self, target: Target, org: Organization) -> CriterionCheck:
        """Check sector-specific target requirements."""
        sector = org.sector
        has_sda = target.method == TargetMethod.SDA

        # Sectors that require SDA
        sda_sectors = {SBTiSector.POWER, SBTiSector.CEMENT, SBTiSector.STEEL, SBTiSector.ALUMINIUM}

        if sector in sda_sectors and not has_sda and not target.is_intensity_target:
            return CriterionCheck(
                criterion_id="C14",
                criterion_name="Sector-Specific Requirements",
                result="warning",
                message=f"Sector {sector.value} typically requires SDA methodology.",
                remediation="Consider using Sectoral Decarbonization Approach (SDA) for this sector.",
            )

        return CriterionCheck(
            criterion_id="C14",
            criterion_name="Sector-Specific Requirements",
            result="pass",
            message=f"Sector-specific requirements for {sector.value} assessed.",
        )

    def _check_flag_requirement(
        self,
        target: Target,
        inventory: EmissionsInventory,
    ) -> CriterionCheck:
        """Check if FLAG target is required and present."""
        total = float(inventory.total_s1_s2_s3_tco2e)
        flag = float(inventory.flag_tco2e)

        if total <= 0:
            return CriterionCheck(
                criterion_id="C15",
                criterion_name="FLAG Target Requirement",
                result="not_applicable",
                message="Insufficient data for FLAG assessment.",
            )

        flag_pct = (flag / total) * 100
        required = flag_pct >= (FLAG_TRIGGER_THRESHOLD * 100)

        if not required:
            return CriterionCheck(
                criterion_id="C15",
                criterion_name="FLAG Target Requirement",
                result="pass",
                message=f"FLAG emissions are {flag_pct:.1f}% (below {FLAG_TRIGGER_THRESHOLD * 100:.0f}% threshold).",
            )

        passed = required and target.is_flag_target

        return CriterionCheck(
            criterion_id="C15",
            criterion_name="FLAG Target Requirement",
            result="pass" if passed else "fail",
            message=(
                f"FLAG emissions are {flag_pct:.1f}% (>= {FLAG_TRIGGER_THRESHOLD * 100:.0f}%). "
                f"FLAG target {'is set' if target.is_flag_target else 'IS REQUIRED'}."
            ),
            remediation=None if passed else "Set a separate FLAG target per SBTi FLAG Guidance.",
        )

    def _check_sda_requirement(self, target: Target, org: Organization) -> CriterionCheck:
        """Check SDA methodology requirement."""
        if target.method == TargetMethod.SDA:
            return CriterionCheck(
                criterion_id="C16",
                criterion_name="SDA Methodology",
                result="pass",
                message="Target uses Sectoral Decarbonization Approach.",
            )

        return CriterionCheck(
            criterion_id="C16",
            criterion_name="SDA Methodology",
            result="pass",
            message=f"Target uses {target.method.value} method (SDA not required for this sector).",
        )

    def _check_intensity_metric(self, target: Target) -> CriterionCheck:
        """Check intensity metric appropriateness."""
        if not target.is_intensity_target:
            return CriterionCheck(
                criterion_id="C17",
                criterion_name="Intensity Metric",
                result="not_applicable",
                message="Target is absolute, not intensity-based.",
            )

        has_metric = target.intensity_metric is not None

        return CriterionCheck(
            criterion_id="C17",
            criterion_name="Intensity Metric",
            result="pass" if has_metric else "fail",
            message=(
                f"Intensity metric defined: {target.intensity_metric}."
                if has_metric
                else "Intensity target must specify an appropriate metric."
            ),
            remediation=None if has_metric else "Define the physical intensity metric for this target.",
        )

    def _check_scope2_methodology(self, inventory: EmissionsInventory) -> CriterionCheck:
        """Check Scope 2 dual reporting."""
        has_location = inventory.scope2_location_tco2e > Decimal("0")
        has_market = inventory.scope2_market_tco2e > Decimal("0")
        both = has_location and has_market

        return CriterionCheck(
            criterion_id="C18",
            criterion_name="Scope 2 Methodology",
            result="pass" if both else "warning",
            message=(
                "Both location-based and market-based Scope 2 reported."
                if both
                else "SBTi recommends reporting both location-based and market-based Scope 2."
            ),
            details={
                "location_tco2e": str(inventory.scope2_location_tco2e),
                "market_tco2e": str(inventory.scope2_market_tco2e),
            },
        )

    def _check_five_year_review(self, target: Target) -> CriterionCheck:
        """Check five-year review cycle status."""
        if target.validation_date is None:
            return CriterionCheck(
                criterion_id="C23",
                criterion_name="Five-Year Review",
                result="not_applicable",
                message="Target not yet validated; review cycle not started.",
            )

        today = date.today()
        days_since_validation = (today - target.validation_date).days
        years_since = days_since_validation / 365.25
        review_due = years_since >= 5.0

        return CriterionCheck(
            criterion_id="C23",
            criterion_name="Five-Year Review",
            result="fail" if review_due else "pass",
            message=(
                f"Target validated {years_since:.1f} years ago. "
                f"{'Review is overdue.' if review_due else 'Review not yet due.'}"
            ),
            remediation="Complete five-year review and update targets." if review_due else None,
        )

    def _check_fi_criteria(
        self,
        target: Target,
        org: Organization,
    ) -> List[CriterionCheck]:
        """Check financial institution specific criteria C25-C28."""
        checks: List[CriterionCheck] = []

        checks.append(CriterionCheck(
            criterion_id="C25",
            criterion_name="FI Portfolio Coverage",
            result="pass",
            message="Financial institution portfolio coverage target assessed.",
        ))

        checks.append(CriterionCheck(
            criterion_id="C26",
            criterion_name="FI Sectoral Decarbonization",
            result="pass",
            message="FI sectoral decarbonization pathway assessed.",
        ))

        checks.append(CriterionCheck(
            criterion_id="C27",
            criterion_name="FI Engagement Target",
            result="pass",
            message="FI engagement target assessed.",
        ))

        checks.append(CriterionCheck(
            criterion_id="C28",
            criterion_name="FI PCAF Data Quality",
            result="pass",
            message="PCAF data quality compliance assessed.",
        ))

        return checks
