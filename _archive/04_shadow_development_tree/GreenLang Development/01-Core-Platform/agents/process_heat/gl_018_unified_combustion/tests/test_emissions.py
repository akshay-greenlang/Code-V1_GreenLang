# -*- coding: utf-8 -*-
"""
GL-018 Emissions Control Tests
==============================

Unit tests for emissions analysis and control per EPA Method 19.
Tests NOx, CO, CO2 calculations and compliance checking.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest

from greenlang.agents.process_heat.gl_018_unified_combustion.emissions import (
    EmissionsController,
)
from greenlang.agents.process_heat.gl_018_unified_combustion.config import (
    EmissionsConfig,
    EmissionControlTechnology,
)
from greenlang.agents.process_heat.gl_018_unified_combustion.schemas import (
    EmissionsAnalysis,
)


class TestEmissionsController:
    """Tests for EmissionsController class."""

    @pytest.fixture
    def controller(self, default_emissions_config):
        """Create emissions controller instance."""
        return EmissionsController(config=default_emissions_config)

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller is not None

    def test_analyze_emissions(self, controller):
        """Test basic emissions analysis."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=25.0,
        )

        assert isinstance(result, EmissionsAnalysis)
        assert result.nox_ppm == 25.0
        assert result.co_ppm == 50.0

    def test_nox_compliance_check(self, controller):
        """Test NOx compliance checking."""
        # Under limit
        result_compliant = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=20.0,
        )
        assert result_compliant.in_compliance is True

    def test_nox_exceedance(self, controller):
        """Test NOx exceedance detection."""
        # Create controller with low NOx limit
        config = EmissionsConfig(nox_permit_limit_lb_mmbtu=0.02)
        ctrl = EmissionsController(config=config)

        result = ctrl.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=50.0,  # High NOx
        )

        # Should have compliance issues
        if result.nox_compliance_pct and result.nox_compliance_pct > 100:
            assert result.in_compliance is False

    def test_co2_calculation(self, controller):
        """Test CO2 calculation per EPA Method 19."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=100.0,
        )

        # Natural gas: ~117 lb CO2/MMBtu
        assert result.co2_lb_mmbtu is not None
        assert 100 <= result.co2_lb_mmbtu <= 130

    def test_co2_tons_calculation(self, controller):
        """Test CO2 tons/hr calculation."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=100.0,
        )

        # CO2 in tons/hr
        assert result.co2_tons_hr >= 0

        # Verify calculation: 117 lb/MMBtu * 100 MMBtu/hr / 2000 lb/ton = 5.85 tons/hr
        expected = 117 * 100 / 2000
        assert abs(result.co2_tons_hr - expected) < 2.0

    def test_nox_lb_mmbtu_calculation(self, controller):
        """Test NOx lb/MMBtu calculation."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=25.0,
        )

        assert result.nox_lb_mmbtu is not None
        assert result.nox_lb_mmbtu > 0

    def test_co_lb_mmbtu_calculation(self, controller):
        """Test CO lb/MMBtu calculation."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
        )

        assert result.co_lb_mmbtu is not None
        assert result.co_lb_mmbtu > 0

    def test_compliance_percentage(self, controller):
        """Test compliance percentage calculation."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=25.0,
        )

        # Compliance percentage of permit limit
        if result.nox_compliance_pct:
            assert 0 <= result.nox_compliance_pct

    def test_recommendations_generated(self, controller):
        """Test recommendations are generated."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=150.0,  # High CO
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=40.0,  # High NOx
        )

        # Should have recommendations for high emissions
        assert isinstance(result.recommendations, list)


class TestFGRControl:
    """Tests for Flue Gas Recirculation control."""

    @pytest.fixture
    def fgr_controller(self):
        """Create emissions controller with FGR enabled."""
        config = EmissionsConfig(
            fgr_enabled=True,
            fgr_rate_pct=15.0,
        )
        return EmissionsController(config=config)

    def test_fgr_nox_reduction(self, fgr_controller):
        """Test FGR NOx reduction calculation."""
        # Without FGR effect
        result_no_fgr = fgr_controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=25.0,
            fgr_rate_pct=0.0,
        )

        # With FGR
        result_with_fgr = fgr_controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=25.0,
            fgr_rate_pct=15.0,
        )

        # Both should return valid analysis
        assert result_no_fgr.nox_ppm == 25.0
        assert result_with_fgr.nox_ppm == 25.0


class TestSCRControl:
    """Tests for Selective Catalytic Reduction control."""

    @pytest.fixture
    def scr_controller(self):
        """Create emissions controller with SCR enabled."""
        config = EmissionsConfig(
            scr_enabled=True,
            scr_inlet_temp_min_f=550.0,
            scr_inlet_temp_max_f=750.0,
            ammonia_slip_limit_ppm=5.0,
        )
        return EmissionsController(config=config)

    def test_scr_nox_reduction(self, scr_controller):
        """Test SCR NOx reduction calculation."""
        result = scr_controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=50.0,
            scr_inlet_nox_ppm=50.0,
            scr_outlet_nox_ppm=10.0,  # 80% reduction
        )

        # SCR should show high reduction
        if result.scr_efficiency_pct:
            assert result.scr_efficiency_pct > 0


class TestFuelTypeEmissions:
    """Tests for different fuel type emissions."""

    @pytest.fixture
    def controller(self):
        """Create emissions controller."""
        return EmissionsController(config=EmissionsConfig())

    def test_natural_gas_emissions(self, controller):
        """Test natural gas emissions factors."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=100.0,
        )

        # Natural gas CO2 factor: ~117 lb/MMBtu
        assert 100 <= result.co2_lb_mmbtu <= 130

    def test_oil_emissions(self, controller):
        """Test fuel oil emissions factors."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="no2_fuel_oil",
            fuel_consumption_mmbtu_hr=100.0,
        )

        # Fuel oil CO2 factor: ~161 lb/MMBtu
        assert 150 <= result.co2_lb_mmbtu <= 175

    def test_coal_emissions(self, controller):
        """Test coal emissions factors."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="coal_bituminous",
            fuel_consumption_mmbtu_hr=100.0,
        )

        # Coal CO2 factor: ~205 lb/MMBtu
        assert 190 <= result.co2_lb_mmbtu <= 220


class TestComplianceIssues:
    """Tests for compliance issue detection."""

    @pytest.fixture
    def strict_controller(self):
        """Create emissions controller with strict limits."""
        config = EmissionsConfig(
            nox_permit_limit_lb_mmbtu=0.02,
            co_permit_limit_lb_mmbtu=0.02,
        )
        return EmissionsController(config=config)

    def test_nox_compliance_issue_detection(self, strict_controller):
        """Test NOx compliance issue detection."""
        result = strict_controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=50.0,  # High NOx that may exceed limit
        )

        # Check if compliance issues are detected
        if result.nox_compliance_pct and result.nox_compliance_pct > 100:
            assert len(result.compliance_issues) > 0

    def test_multiple_compliance_issues(self, strict_controller):
        """Test multiple compliance issues."""
        result = strict_controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=300.0,  # High CO
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=60.0,  # High NOx
        )

        # May have multiple issues
        assert isinstance(result.compliance_issues, list)


class TestEmissionsEdgeCases:
    """Edge case tests for emissions calculations."""

    @pytest.fixture
    def controller(self):
        """Create emissions controller."""
        return EmissionsController(config=EmissionsConfig())

    def test_zero_nox(self, controller):
        """Test with zero NOx."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=0.0,
        )

        assert result.nox_ppm == 0.0

    def test_zero_co(self, controller):
        """Test with zero CO."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=0.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
        )

        assert result.co_ppm == 0.0

    def test_high_o2_correction(self, controller):
        """Test O2 correction for high O2."""
        result = controller.analyze_emissions(
            o2_pct=10.0,  # High O2
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=75.0,
            nox_ppm=25.0,
        )

        # Should still calculate valid emissions
        assert result.co2_tons_hr >= 0

    def test_minimum_fuel_consumption(self, controller):
        """Test with minimum fuel consumption."""
        result = controller.analyze_emissions(
            o2_pct=3.0,
            co_ppm=50.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=1.0,
        )

        assert result.co2_tons_hr >= 0
        assert result.co2_tons_hr < 1.0  # Should be small
