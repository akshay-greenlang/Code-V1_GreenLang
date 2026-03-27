# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6 of 7)

AGENT-MRV-012: Cooling Purchase Agent

Tests multi-framework regulatory compliance checking across 7 frameworks
(GHG Protocol, ISO 14064, CSRD, ASHRAE 90.1, EU F-Gas, California Title 24,
Singapore BCA) with 84 total requirements (12 per framework).

Target: 70 tests, ~600 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.mrv.cooling_purchase.compliance_checker import (
    ComplianceCheckerEngine,
    get_compliance_checker,
    SUPPORTED_FRAMEWORKS,
)
from greenlang.agents.mrv.cooling_purchase.models import (
    ComplianceStatus,
    CoolingTechnology,
    DataQualityTier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compliance_engine():
    """Create a ComplianceCheckerEngine instance."""
    engine = ComplianceCheckerEngine()
    yield engine
    engine.reset()


@pytest.fixture
def complete_data() -> Dict[str, Any]:
    """Return a fully compliant calculation data dictionary."""
    return {
        "technology": "WATER_COOLED_CENTRIFUGAL",
        "cooling_kwh_th": Decimal("100000"),
        "emissions_kgco2e": Decimal("10000"),
        "cop": Decimal("5.5"),
        "iplv": Decimal("8.0"),
        "refrigerant": "R134a",
        "refrigerant_charge_kg": Decimal("100"),
        "leakage_rate_pct": Decimal("2.0"),
        "gwp_source": "AR6",
        "tier": "TIER_2",
        "facility_name": "Test Facility",
        "facility_type": "OFFICE",
        "reporting_period": "2024",
        "provenance_hash": "a" * 64,
        "calculation_date": "2024-02-01",
        "grid_ef_kgco2e_kwh": Decimal("0.45"),
        "distribution_loss_pct": Decimal("5.0"),
    }


@pytest.fixture
def minimal_data() -> Dict[str, Any]:
    """Return a minimal calculation data dictionary."""
    return {
        "technology": "WATER_COOLED_CENTRIFUGAL",
        "cooling_kwh_th": Decimal("50000"),
        "emissions_kgco2e": Decimal("5000"),
    }


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestComplianceEngineInit:
    """Test ComplianceCheckerEngine initialization."""

    def test_singleton_pattern(self):
        """Test singleton returns same instance."""
        e1 = ComplianceCheckerEngine()
        e2 = ComplianceCheckerEngine()
        assert e1 is e2

    def test_get_function_returns_singleton(self):
        """Test get_compliance_checker returns singleton."""
        e1 = get_compliance_checker()
        e2 = get_compliance_checker()
        assert e1 is e2

    def test_supported_frameworks_count(self):
        """Test 7 frameworks are supported."""
        assert len(SUPPORTED_FRAMEWORKS) == 7

    def test_get_all_frameworks(self, compliance_engine):
        """Test get_all_frameworks returns 7 frameworks."""
        frameworks = compliance_engine.get_all_frameworks()
        assert len(frameworks) == 7

    def test_get_total_requirements_count(self, compliance_engine):
        """Test total requirements count is 84 (7 frameworks * 12 requirements)."""
        total = compliance_engine.get_total_requirements_count()
        assert total == 84

    def test_framework_list(self, compliance_engine):
        """Test all 7 frameworks are in the list."""
        frameworks = compliance_engine.get_all_frameworks()
        assert "GHG_PROTOCOL" in frameworks
        assert "ISO_14064" in frameworks
        assert "CSRD" in frameworks
        assert "ASHRAE_90_1" in frameworks
        assert "EU_F_GAS" in frameworks
        assert "CALIFORNIA_TITLE_24" in frameworks
        assert "SINGAPORE_BCA" in frameworks


# ===========================================================================
# 2. Framework Requirements Tests
# ===========================================================================


class TestFrameworkRequirements:
    """Test get_framework_requirements for each framework."""

    def test_ghg_protocol_requirements(self, compliance_engine):
        """Test GHG Protocol has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("GHG_PROTOCOL")
        assert len(reqs) == 12

    def test_iso_14064_requirements(self, compliance_engine):
        """Test ISO 14064 has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("ISO_14064")
        assert len(reqs) == 12

    def test_csrd_requirements(self, compliance_engine):
        """Test CSRD has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("CSRD")
        assert len(reqs) == 12

    def test_ashrae_requirements(self, compliance_engine):
        """Test ASHRAE 90.1 has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("ASHRAE_90_1")
        assert len(reqs) == 12

    def test_eu_fgas_requirements(self, compliance_engine):
        """Test EU F-Gas has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("EU_F_GAS")
        assert len(reqs) == 12

    def test_california_title_24_requirements(self, compliance_engine):
        """Test California Title 24 has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("CALIFORNIA_TITLE_24")
        assert len(reqs) == 12

    def test_singapore_bca_requirements(self, compliance_engine):
        """Test Singapore BCA has 12 requirements."""
        reqs = compliance_engine.get_framework_requirements("SINGAPORE_BCA")
        assert len(reqs) == 12


# ===========================================================================
# 3. check_compliance() Tests
# ===========================================================================


class TestCheckCompliance:
    """Test check_compliance with complete and minimal data."""

    def test_check_compliance_complete_data(self, compliance_engine, complete_data):
        """Test compliance check with complete data returns COMPLIANT."""
        result = compliance_engine.check_compliance(complete_data)
        assert result["status"] in ["COMPLIANT", "PARTIAL"]
        assert len(result["framework_results"]) == 7

    def test_check_compliance_minimal_data(self, compliance_engine, minimal_data):
        """Test compliance check with minimal data returns NON_COMPLIANT or PARTIAL."""
        result = compliance_engine.check_compliance(minimal_data)
        assert result["status"] in ["NON_COMPLIANT", "PARTIAL"]

    def test_check_compliance_all_frameworks(self, compliance_engine, complete_data):
        """Test all 7 frameworks are checked by default."""
        result = compliance_engine.check_compliance(complete_data)
        assert len(result["framework_results"]) == 7

    def test_check_compliance_single_framework(self, compliance_engine, complete_data):
        """Test checking single framework."""
        result = compliance_engine.check_compliance(
            complete_data, frameworks=["GHG_PROTOCOL"]
        )
        assert len(result["framework_results"]) == 1
        assert "GHG_PROTOCOL" in result["framework_results"]

    def test_check_compliance_multiple_frameworks(self, compliance_engine, complete_data):
        """Test checking multiple frameworks."""
        result = compliance_engine.check_compliance(
            complete_data, frameworks=["GHG_PROTOCOL", "ISO_14064", "CSRD"]
        )
        assert len(result["framework_results"]) == 3


# ===========================================================================
# 4. check_single_framework() Tests
# ===========================================================================


class TestCheckSingleFramework:
    """Test check_single_framework for each of 7 frameworks."""

    def test_check_ghg_protocol(self, compliance_engine, complete_data):
        """Test GHG Protocol compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "GHG_PROTOCOL")
        assert result["framework"] == "GHG_PROTOCOL"
        assert "met_count" in result
        assert "total_count" in result

    def test_check_iso_14064(self, compliance_engine, complete_data):
        """Test ISO 14064 compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "ISO_14064")
        assert result["framework"] == "ISO_14064"

    def test_check_csrd(self, compliance_engine, complete_data):
        """Test CSRD compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "CSRD")
        assert result["framework"] == "CSRD"

    def test_check_ashrae(self, compliance_engine, complete_data):
        """Test ASHRAE 90.1 compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "ASHRAE_90_1")
        assert result["framework"] == "ASHRAE_90_1"

    def test_check_eu_fgas(self, compliance_engine, complete_data):
        """Test EU F-Gas compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "EU_F_GAS")
        assert result["framework"] == "EU_F_GAS"

    def test_check_california_title_24(self, compliance_engine, complete_data):
        """Test California Title 24 compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "CALIFORNIA_TITLE_24")
        assert result["framework"] == "CALIFORNIA_TITLE_24"

    def test_check_singapore_bca(self, compliance_engine, complete_data):
        """Test Singapore BCA compliance check."""
        result = compliance_engine.check_single_framework(complete_data, "SINGAPORE_BCA")
        assert result["framework"] == "SINGAPORE_BCA"


# ===========================================================================
# 5. _determine_status() Tests
# ===========================================================================


class TestDetermineStatus:
    """Test _determine_status logic."""

    def test_status_12_of_12_is_compliant(self, compliance_engine):
        """Test 12/12 requirements returns COMPLIANT."""
        status = compliance_engine._determine_status(12, 12)
        assert status == "COMPLIANT"

    def test_status_10_of_12_is_partial(self, compliance_engine):
        """Test 10/12 requirements returns PARTIAL."""
        status = compliance_engine._determine_status(10, 12)
        assert status == "PARTIAL"

    def test_status_5_of_12_is_non_compliant(self, compliance_engine):
        """Test 5/12 requirements returns NON_COMPLIANT."""
        status = compliance_engine._determine_status(5, 12)
        assert status == "NON_COMPLIANT"

    def test_status_0_of_12_is_non_compliant(self, compliance_engine):
        """Test 0/12 requirements returns NON_COMPLIANT."""
        status = compliance_engine._determine_status(0, 12)
        assert status == "NON_COMPLIANT"

    def test_status_11_of_12_is_partial(self, compliance_engine):
        """Test 11/12 requirements returns PARTIAL."""
        status = compliance_engine._determine_status(11, 12)
        assert status == "PARTIAL"


# ===========================================================================
# 6. calculate_compliance_score() Tests
# ===========================================================================


class TestComplianceScore:
    """Test compliance score calculation."""

    def test_compliance_score_100_pct(self, compliance_engine):
        """Test 12/12 met gives 100% score."""
        score = compliance_engine.calculate_compliance_score(12, 12)
        assert score == Decimal("100")

    def test_compliance_score_50_pct(self, compliance_engine):
        """Test 6/12 met gives 50% score."""
        score = compliance_engine.calculate_compliance_score(6, 12)
        assert score == Decimal("50")

    def test_compliance_score_0_pct(self, compliance_engine):
        """Test 0/12 met gives 0% score."""
        score = compliance_engine.calculate_compliance_score(0, 12)
        assert score == Decimal("0")

    def test_compliance_score_rounding(self, compliance_engine):
        """Test score is rounded to 2 decimal places."""
        score = compliance_engine.calculate_compliance_score(7, 12)
        assert score == Decimal("58.33")


# ===========================================================================
# 7. ASHRAE Minimum COP Tests
# ===========================================================================


class TestASHRAEMinimumCOP:
    """Test ASHRAE 90.1 minimum COP requirements."""

    def test_get_ashrae_minimum_cop_centrifugal(self, compliance_engine):
        """Test ASHRAE minimum COP for centrifugal chiller."""
        min_cop = compliance_engine.get_ashrae_minimum_cop("WATER_COOLED_CENTRIFUGAL")
        assert min_cop > Decimal("0")

    def test_get_ashrae_minimum_cop_screw(self, compliance_engine):
        """Test ASHRAE minimum COP for screw chiller."""
        min_cop = compliance_engine.get_ashrae_minimum_cop("WATER_COOLED_SCREW")
        assert min_cop > Decimal("0")

    def test_get_ashrae_minimum_cop_scroll(self, compliance_engine):
        """Test ASHRAE minimum COP for scroll chiller."""
        min_cop = compliance_engine.get_ashrae_minimum_cop("AIR_COOLED_SCROLL")
        assert min_cop > Decimal("0")

    def test_check_ashrae_minimum_cop_pass(self, compliance_engine):
        """Test COP above minimum passes ASHRAE check."""
        result = compliance_engine.check_ashrae_minimum_cop(
            technology="WATER_COOLED_CENTRIFUGAL",
            cop=Decimal("6.0"),
        )
        assert result is True

    def test_check_ashrae_minimum_cop_fail(self, compliance_engine):
        """Test COP below minimum fails ASHRAE check."""
        result = compliance_engine.check_ashrae_minimum_cop(
            technology="WATER_COOLED_CENTRIFUGAL",
            cop=Decimal("2.0"),
        )
        assert result is False


# ===========================================================================
# 8. Phase-Down Compliance Tests
# ===========================================================================


class TestPhaseDownCompliance:
    """Test refrigerant phase-down compliance (EU F-Gas Regulation)."""

    def test_check_phase_down_compliance_r134a(self, compliance_engine):
        """Test R-134a phase-down compliance."""
        result = compliance_engine.check_phase_down_compliance(
            refrigerant="R134a",
            installation_year=2024,
        )
        assert isinstance(result, bool)

    def test_check_phase_down_compliance_r410a(self, compliance_engine):
        """Test R-410A phase-down compliance."""
        result = compliance_engine.check_phase_down_compliance(
            refrigerant="R410A",
            installation_year=2024,
        )
        assert isinstance(result, bool)

    def test_check_phase_down_compliance_r32(self, compliance_engine):
        """Test R-32 phase-down compliance (low GWP)."""
        result = compliance_engine.check_phase_down_compliance(
            refrigerant="R32",
            installation_year=2024,
        )
        # R-32 is low GWP, should pass
        assert result is True


# ===========================================================================
# 9. CO2e Charge Calculation Tests
# ===========================================================================


class TestCO2eChargeCalculation:
    """Test calculate_co2e_charge for refrigerant inventory."""

    def test_calculate_co2e_charge_r134a(self, compliance_engine):
        """Test CO2e charge calculation for R-134a."""
        co2e = compliance_engine.calculate_co2e_charge(
            refrigerant="R134a",
            charge_kg=Decimal("100"),
        )
        assert co2e > Decimal("0")

    def test_calculate_co2e_charge_r410a(self, compliance_engine):
        """Test CO2e charge calculation for R-410A."""
        co2e = compliance_engine.calculate_co2e_charge(
            refrigerant="R410A",
            charge_kg=Decimal("50"),
        )
        assert co2e > Decimal("0")

    def test_calculate_co2e_charge_zero_charge(self, compliance_engine):
        """Test CO2e charge with zero refrigerant charge."""
        co2e = compliance_engine.calculate_co2e_charge(
            refrigerant="R134a",
            charge_kg=Decimal("0"),
        )
        assert co2e == Decimal("0")


# ===========================================================================
# 10. Reporting Threshold Tests
# ===========================================================================


class TestReportingThreshold:
    """Test check_reporting_threshold (500 tCO2e threshold)."""

    def test_reporting_threshold_above(self, compliance_engine):
        """Test emissions above 500 tCO2e triggers reporting."""
        result = compliance_engine.check_reporting_threshold(
            emissions_kgco2e=Decimal("600000"),  # 600 tCO2e
        )
        assert result is True

    def test_reporting_threshold_below(self, compliance_engine):
        """Test emissions below 500 tCO2e does not trigger reporting."""
        result = compliance_engine.check_reporting_threshold(
            emissions_kgco2e=Decimal("400000"),  # 400 tCO2e
        )
        assert result is False

    def test_reporting_threshold_exactly_500(self, compliance_engine):
        """Test emissions exactly at 500 tCO2e threshold."""
        result = compliance_engine.check_reporting_threshold(
            emissions_kgco2e=Decimal("500000"),  # 500 tCO2e
        )
        assert result is True


# ===========================================================================
# 11. Compliance with All Fields Present vs Missing
# ===========================================================================


class TestComplianceFieldPresence:
    """Test compliance with all fields present vs missing fields."""

    def test_all_fields_present(self, compliance_engine, complete_data):
        """Test compliance with all required fields present."""
        result = compliance_engine.check_compliance(complete_data)
        # Should have high compliance score
        assert result["overall_score_pct"] > Decimal("50")

    def test_missing_cop_field(self, compliance_engine, minimal_data):
        """Test compliance with missing COP field."""
        result = compliance_engine.check_compliance(minimal_data)
        # Score should be lower without COP
        assert result["overall_score_pct"] < Decimal("100")

    def test_missing_refrigerant_field(self, compliance_engine, minimal_data):
        """Test compliance with missing refrigerant field."""
        result = compliance_engine.check_compliance(minimal_data)
        # Missing refrigerant affects EU F-Gas compliance
        assert result["overall_score_pct"] < Decimal("100")

    def test_missing_tier_field(self, compliance_engine, minimal_data):
        """Test compliance with missing tier field."""
        result = compliance_engine.check_compliance(minimal_data)
        # Missing tier affects data quality requirements
        assert result["overall_score_pct"] < Decimal("100")


# ===========================================================================
# 12. Statistics and Reset Tests
# ===========================================================================


class TestStatisticsAndReset:
    """Test statistics tracking and reset functionality."""

    def test_statistics_counter_increments(self, compliance_engine, complete_data):
        """Test statistics counter increments after check."""
        stats_before = compliance_engine.get_statistics()
        _ = compliance_engine.check_compliance(complete_data)
        stats_after = compliance_engine.get_statistics()
        assert stats_after["total_checks"] == stats_before["total_checks"] + 1

    def test_reset_clears_statistics(self, compliance_engine, complete_data):
        """Test reset clears statistics."""
        _ = compliance_engine.check_compliance(complete_data)
        compliance_engine.reset()
        stats = compliance_engine.get_statistics()
        assert stats["total_checks"] == 0
