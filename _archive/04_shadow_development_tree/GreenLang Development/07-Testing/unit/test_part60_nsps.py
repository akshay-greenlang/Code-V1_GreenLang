"""
Unit tests for EPA Part 60 NSPS Compliance Checker.

Tests cover:
    - F-factor calculations (Fd, Fc, Fw)
    - Subpart compliance checking (D, Db, Dc, J)
    - Emission limit calculations
    - Compliance report generation
    - Provenance tracking
"""

import pytest
from datetime import datetime, timezone

from greenlang.compliance.epa import (
    NSPSComplianceChecker,
    FuelType,
    BoilerType,
    ComplianceStatus,
    EmissionsData,
    FacilityData,
    FFactorCalculator,
)


# =============================================================================
# F-FACTOR CALCULATOR TESTS
# =============================================================================


class TestFFactorCalculator:
    """Test F-factor calculations per EPA Method 19."""

    def test_calculate_fd_natural_gas(self):
        """Test Fd calculation for natural gas (no sulfur)."""
        fd = FFactorCalculator.calculate_fd("natural_gas", so2_fraction=0.0)
        assert 8.0 < fd < 10.0  # Natural gas Fd typically 9.0-9.2
        assert fd > 0

    def test_calculate_fd_coal_with_sulfur(self):
        """Test Fd calculation for coal with sulfur content."""
        # Coal with 2% sulfur
        fd = FFactorCalculator.calculate_fd("coal_bituminous", so2_fraction=0.02)
        assert fd > 0
        # Fd should be less than base factor due to sulfur
        fd_no_sulfur = FFactorCalculator.calculate_fd("coal_bituminous", so2_fraction=0.0)
        assert fd < fd_no_sulfur

    def test_calculate_fd_high_sulfur_coal(self):
        """Test Fd with high sulfur coal."""
        fd = FFactorCalculator.calculate_fd("coal_bituminous", so2_fraction=0.05)
        assert fd >= 1.0  # Minimum bound

    def test_calculate_fc_standard_oxygen(self):
        """Test Fc correction factor at standard 3% O2."""
        fc = FFactorCalculator.calculate_fc(excess_o2_pct=3.0)
        assert abs(fc - 1.0) < 0.01  # Should be ~1.0 at reference

    def test_calculate_fc_excess_oxygen(self):
        """Test Fc with excess oxygen."""
        fc_high = FFactorCalculator.calculate_fc(excess_o2_pct=5.0)
        fc_ref = FFactorCalculator.calculate_fc(excess_o2_pct=3.0)
        assert fc_high < fc_ref  # Higher O2 = lower Fc

    def test_calculate_fw_coal_moisture(self):
        """Test Fw moisture correction for coal."""
        fw = FFactorCalculator.calculate_fw("coal_bituminous", moisture_pct=5.0)
        assert 0.8 < fw < 1.2

    def test_calculate_fw_natural_gas_dry(self):
        """Test Fw for dry natural gas."""
        fw = FFactorCalculator.calculate_fw("natural_gas", moisture_pct=0.0)
        assert fw > 0.8


# =============================================================================
# SUBPART D TESTS (Fossil-Fuel-Fired Steam Generators >100 MMBtu/hr)
# =============================================================================


class TestSubpartD:
    """Test Subpart D compliance checking."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    @pytest.fixture
    def facility_natural_gas(self):
        """Create test facility with natural gas, >100 MMBtu/hr."""
        return FacilityData(
            facility_id="FAC-001",
            equipment_id="BOILER-001",
            boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=150.0,
            permit_limits={"SO2": 0.020, "NOx": 0.50},
        )

    @pytest.fixture
    def emissions_compliant(self):
        """Create compliant emissions data."""
        return EmissionsData(
            so2_lb_mmbtu=0.015,  # Below 0.020 limit
            nox_lb_mmbtu=0.45,   # Below 0.50 limit
            pm_gr_dscf=0.020,
            opacity_pct=15.0,    # Below 20% limit
            o2_pct=3.5,
        )

    @pytest.fixture
    def emissions_noncompliant(self):
        """Create non-compliant emissions data."""
        return EmissionsData(
            so2_lb_mmbtu=0.035,  # Above 0.020 limit
            nox_lb_mmbtu=0.60,   # Above 0.50 limit
            pm_gr_dscf=0.035,
            opacity_pct=25.0,    # Above 20% limit
            o2_pct=3.5,
        )

    def test_subpart_d_compliant(self, checker, facility_natural_gas, emissions_compliant):
        """Test Subpart D with compliant emissions."""
        result = checker.check_subpart_d(facility_natural_gas, emissions_compliant)

        assert result.compliance_status == ComplianceStatus.COMPLIANT
        assert result.so2_status == "PASS"
        assert result.nox_status == "PASS"
        assert result.opacity_status == "PASS"
        assert result.so2_compliance_margin > 0
        assert result.nox_compliance_margin > 0

    def test_subpart_d_noncompliant_so2(self, checker, facility_natural_gas, emissions_noncompliant):
        """Test Subpart D with SO2 exceedance."""
        result = checker.check_subpart_d(facility_natural_gas, emissions_noncompliant)

        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert result.so2_status == "FAIL"
        assert result.so2_compliance_margin < 0
        assert any("SO2 EXCEEDANCE" in f for f in result.findings)

    def test_subpart_d_noncompliant_nox(self, checker, facility_natural_gas, emissions_noncompliant):
        """Test Subpart D with NOx exceedance."""
        result = checker.check_subpart_d(facility_natural_gas, emissions_noncompliant)

        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert result.nox_status == "FAIL"
        assert result.nox_compliance_margin < 0
        assert any("NOx EXCEEDANCE" in f for f in result.findings)

    def test_subpart_d_coal_higher_limit(self, checker):
        """Test Subpart D with coal fuel (higher SO2 limit)."""
        facility_coal = FacilityData(
            facility_id="FAC-002",
            equipment_id="BOILER-002",
            boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
            fuel_type=FuelType.COAL,
            heat_input_mmbtu_hr=150.0,
        )
        emissions = EmissionsData(
            so2_lb_mmbtu=0.25,   # Below 0.30 limit for coal
            nox_lb_mmbtu=0.45,
            o2_pct=3.5,
        )
        result = checker.check_subpart_d(facility_coal, emissions)

        assert result.so2_status == "PASS"
        assert result.so2_limit_lb_mmbtu == 0.30

    def test_subpart_d_provenance_hash(self, checker, facility_natural_gas, emissions_compliant):
        """Test provenance hash generation."""
        result = checker.check_subpart_d(facility_natural_gas, emissions_compliant)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex digest


# =============================================================================
# SUBPART Db TESTS (Industrial Boilers 10-100 MMBtu/hr)
# =============================================================================


class TestSubpartDb:
    """Test Subpart Db compliance checking."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    @pytest.fixture
    def facility_industrial_boiler(self):
        """Create test facility 10-100 MMBtu/hr."""
        return FacilityData(
            facility_id="FAC-003",
            equipment_id="BOILER-003",
            boiler_type=BoilerType.INDUSTRIAL_BOILER,
            fuel_type=FuelType.DISTILLATE_OIL,
            heat_input_mmbtu_hr=50.0,
        )

    def test_subpart_db_nox_fuel_specific(self, checker, facility_industrial_boiler):
        """Test that Subpart Db uses fuel-specific NOx limits."""
        emissions_gas = EmissionsData(
            nox_lb_mmbtu=0.055,   # Below 0.060 for natural gas
            o2_pct=3.5,
        )

        facility_gas = FacilityData(
            facility_id="FAC-004",
            equipment_id="BOILER-004",
            boiler_type=BoilerType.INDUSTRIAL_BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=50.0,
        )

        result = checker.check_subpart_db(facility_gas, emissions_gas)
        assert result.nox_limit_lb_mmbtu == 0.060
        assert result.nox_status == "PASS"

    def test_subpart_db_coal_nox(self, checker):
        """Test Subpart Db with coal (NOx limit 0.30)."""
        facility = FacilityData(
            facility_id="FAC-005",
            equipment_id="BOILER-005",
            boiler_type=BoilerType.INDUSTRIAL_BOILER,
            fuel_type=FuelType.COAL,
            heat_input_mmbtu_hr=50.0,
        )
        emissions = EmissionsData(
            nox_lb_mmbtu=0.25,
            o2_pct=3.5,
        )
        result = checker.check_subpart_db(facility, emissions)

        assert result.nox_limit_lb_mmbtu == 0.30
        assert result.nox_status == "PASS"


# =============================================================================
# SUBPART Dc TESTS (Small Boilers <10 MMBtu/hr)
# =============================================================================


class TestSubpartDc:
    """Test Subpart Dc compliance checking."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    @pytest.fixture
    def facility_small_boiler(self):
        """Create test facility <10 MMBtu/hr."""
        return FacilityData(
            facility_id="FAC-006",
            equipment_id="BOILER-006",
            boiler_type=BoilerType.SMALL_BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=5.0,
        )

    def test_subpart_dc_relaxed_limits(self, checker, facility_small_boiler):
        """Test that Subpart Dc has more relaxed limits than Db."""
        emissions = EmissionsData(
            so2_lb_mmbtu=0.025,   # Between Db and Dc limits
            nox_lb_mmbtu=0.075,
            o2_pct=3.5,
        )
        result = checker.check_subpart_dc(facility_small_boiler, emissions)

        # Subpart Dc has SO2 limit 0.030 (more relaxed than Db 0.020)
        assert result.so2_limit_lb_mmbtu == 0.030
        assert result.so2_status == "PASS"


# =============================================================================
# SUBPART J TESTS (Petroleum Refineries)
# =============================================================================


class TestSubpartJ:
    """Test Subpart J compliance checking."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    @pytest.fixture
    def facility_refinery(self):
        """Create test refinery furnace."""
        return FacilityData(
            facility_id="FAC-007",
            equipment_id="FURNACE-001",
            boiler_type=BoilerType.PROCESS_HEATER,
            fuel_type=FuelType.COAL_DERIVED,  # Fuel gas at refinery
            heat_input_mmbtu_hr=75.0,
        )

    def test_subpart_j_nox_limit(self, checker, facility_refinery):
        """Test Subpart J NOx limit (0.30 lb/MMBtu)."""
        emissions = EmissionsData(
            nox_lb_mmbtu=0.25,
            o2_pct=3.5,
        )
        result = checker.check_subpart_j(facility_refinery, emissions)

        assert result.nox_limit_lb_mmbtu == 0.30
        assert result.nox_status == "PASS"

    def test_subpart_j_stricter_opacity(self, checker, facility_refinery):
        """Test Subpart J stricter opacity limit (5% vs 20%)."""
        emissions = EmissionsData(
            opacity_pct=4.0,
            o2_pct=3.5,
        )
        result = checker.check_subpart_j(facility_refinery, emissions)

        assert result.opacity_limit_pct == 5.0
        assert result.opacity_status == "PASS"

    def test_subpart_j_co_monitoring(self, checker, facility_refinery):
        """Test Subpart J CO monitoring (guideline not limit)."""
        emissions = EmissionsData(
            co_lb_mmbtu=0.70,    # Above 0.60 guideline
            o2_pct=3.5,
        )
        result = checker.check_subpart_j(facility_refinery, emissions)

        assert any("CO" in f for f in result.findings)


# =============================================================================
# EMISSION LIMIT CALCULATION TESTS
# =============================================================================


class TestEmissionLimitCalculation:
    """Test emission limit calculation method."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    def test_calculate_limits_subpart_d(self, checker):
        """Test limit calculation for Subpart D."""
        limits = checker.calculate_emission_limits(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=150.0,
            subpart="D",
        )

        assert limits["SO2"] == 0.020
        assert limits["NOx"] == 0.50
        assert limits["PM"] == 0.03
        assert limits["Opacity"] == 20.0

    def test_calculate_limits_subpart_db(self, checker):
        """Test limit calculation for Subpart Db."""
        limits = checker.calculate_emission_limits(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=50.0,
            subpart="Db",
        )

        assert limits["SO2"] == 0.020
        assert limits["NOx"] == 0.060
        assert limits["PM"] == 0.015
        assert limits["Opacity"] == 20.0

    def test_calculate_limits_coal_fuel(self, checker):
        """Test limit calculation with coal fuel."""
        limits = checker.calculate_emission_limits(
            fuel_type=FuelType.COAL,
            heat_input_mmbtu_hr=150.0,
            subpart="D",
        )

        assert limits["SO2"] == 0.30  # Coal has higher limit


# =============================================================================
# COMPLIANCE REPORT TESTS
# =============================================================================


class TestComplianceReport:
    """Test comprehensive compliance report generation."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    def test_generate_compliance_report_compliant(self, checker):
        """Test report generation for compliant facility."""
        facility = FacilityData(
            facility_id="FAC-008",
            equipment_id="BOILER-008",
            boiler_type=BoilerType.INDUSTRIAL_BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=50.0,
        )
        emissions = EmissionsData(
            so2_lb_mmbtu=0.015,
            nox_lb_mmbtu=0.055,
            pm_lb_mmbtu=0.012,
            opacity_pct=15.0,
            o2_pct=3.5,
        )

        report = checker.generate_compliance_report(facility, emissions)

        assert report["compliance_status"] == ComplianceStatus.COMPLIANT.value
        assert report["fuel_type"] == FuelType.NATURAL_GAS.value
        assert report["heat_input_mmbtu_hr"] == 50.0
        assert "so2_compliance" in report
        assert "nox_compliance" in report
        assert "pm_compliance" in report
        assert "recommendations" in report

    def test_report_includes_provenance(self, checker):
        """Test that report includes provenance hash."""
        facility = FacilityData(
            facility_id="FAC-009",
            equipment_id="BOILER-009",
            boiler_type=BoilerType.SMALL_BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=5.0,
        )
        emissions = EmissionsData(
            so2_lb_mmbtu=0.020,
            o2_pct=3.5,
        )

        report = checker.generate_compliance_report(facility, emissions)

        assert report["provenance_hash"] is not None
        assert len(report["provenance_hash"]) == 64


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def checker(self):
        """Create NSPSComplianceChecker instance."""
        return NSPSComplianceChecker()

    def test_emissions_at_limit(self, checker):
        """Test emissions exactly at limit."""
        facility = FacilityData(
            facility_id="FAC-010",
            equipment_id="BOILER-010",
            boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=150.0,
        )
        emissions = EmissionsData(
            so2_lb_mmbtu=0.020,  # Exactly at limit
            nox_lb_mmbtu=0.50,   # Exactly at limit
            o2_pct=3.5,
        )

        result = checker.check_subpart_d(facility, emissions)

        assert result.so2_status == "PASS"
        assert result.nox_status == "PASS"

    def test_emissions_just_above_limit(self, checker):
        """Test emissions just above limit."""
        facility = FacilityData(
            facility_id="FAC-011",
            equipment_id="BOILER-011",
            boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=150.0,
        )
        emissions = EmissionsData(
            so2_lb_mmbtu=0.0201,  # Just above limit
            o2_pct=3.5,
        )

        result = checker.check_subpart_d(facility, emissions)

        assert result.so2_status == "FAIL"
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT

    def test_optional_emissions_data(self, checker):
        """Test with missing optional emissions parameters."""
        facility = FacilityData(
            facility_id="FAC-012",
            equipment_id="BOILER-012",
            boiler_type=BoilerType.INDUSTRIAL_BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=50.0,
        )
        # Only SO2 provided, others None
        emissions = EmissionsData(
            so2_lb_mmbtu=0.015,
            nox_lb_mmbtu=None,
            pm_lb_mmbtu=None,
            opacity_pct=None,
            o2_pct=3.5,
        )

        result = checker.check_subpart_db(facility, emissions)

        assert result.so2_status == "PASS"
        assert result.nox_status is None
        assert result.pm_status is None
        assert result.opacity_status is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete compliance workflow."""

    def test_multi_facility_comparison(self):
        """Test compliance checking across multiple facilities."""
        checker = NSPSComplianceChecker()

        facilities = [
            FacilityData(
                facility_id="FAC-A",
                equipment_id="BOILER-A",
                boiler_type=BoilerType.INDUSTRIAL_BOILER,
                fuel_type=FuelType.NATURAL_GAS,
                heat_input_mmbtu_hr=50.0,
            ),
            FacilityData(
                facility_id="FAC-B",
                equipment_id="BOILER-B",
                boiler_type=BoilerType.SMALL_BOILER,
                fuel_type=FuelType.NATURAL_GAS,
                heat_input_mmbtu_hr=5.0,
            ),
        ]

        emissions_list = [
            EmissionsData(so2_lb_mmbtu=0.015, nox_lb_mmbtu=0.055, o2_pct=3.5),
            EmissionsData(so2_lb_mmbtu=0.025, nox_lb_mmbtu=0.075, o2_pct=3.5),
        ]

        results = []
        for fac, emis in zip(facilities, emissions_list):
            if fac.boiler_type == BoilerType.INDUSTRIAL_BOILER:
                result = checker.check_subpart_db(fac, emis)
            else:
                result = checker.check_subpart_dc(fac, emis)
            results.append(result)

        assert len(results) == 2
        assert all(r.facility_id in ["FAC-A", "FAC-B"] for r in results)

    def test_processing_time_tracking(self):
        """Test that processing time is tracked."""
        checker = NSPSComplianceChecker()

        facility = FacilityData(
            facility_id="FAC-013",
            equipment_id="BOILER-013",
            boiler_type=BoilerType.FOSSIL_FUEL_STEAM,
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu_hr=150.0,
        )
        emissions = EmissionsData(
            so2_lb_mmbtu=0.015,
            nox_lb_mmbtu=0.45,
            o2_pct=3.5,
        )

        result = checker.check_subpart_d(facility, emissions)

        assert result.processing_time_ms >= 0  # Should be tracked
        assert result.processing_time_ms < 1000  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
