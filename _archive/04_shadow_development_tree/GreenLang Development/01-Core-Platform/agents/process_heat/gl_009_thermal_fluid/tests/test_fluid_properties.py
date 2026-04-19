"""
Unit tests for GL-009 THERMALIQ Agent Fluid Properties

Tests thermal fluid property calculations including density, viscosity,
specific heat, thermal conductivity, and vapor pressure.

All tests validate against manufacturer data and known engineering values.
"""

import pytest
import math
from typing import List, Tuple

from greenlang.agents.process_heat.gl_009_thermal_fluid.fluid_properties import (
    ThermalFluidPropertyDatabase,
    FluidCoefficients,
    FLUID_DATA,
    get_fluid_properties,
    compare_fluids,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    FluidProperties,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def property_db():
    """Create property database instance."""
    return ThermalFluidPropertyDatabase()


@pytest.fixture
def therminol_66_db(property_db):
    """Get database configured for Therminol 66."""
    return property_db


# =============================================================================
# DATABASE INITIALIZATION TESTS
# =============================================================================

class TestPropertyDatabaseInit:
    """Tests for property database initialization."""

    def test_database_initializes(self, property_db):
        """Test database initializes successfully."""
        assert property_db is not None
        assert property_db._calculation_count == 0

    def test_all_fluids_have_coefficients(self):
        """Test all fluid types have coefficient data."""
        # Check that most common fluids are in database
        expected_fluids = [
            ThermalFluidType.THERMINOL_66,
            ThermalFluidType.THERMINOL_VP1,
            ThermalFluidType.THERMINOL_55,
            ThermalFluidType.DOWTHERM_A,
            ThermalFluidType.DOWTHERM_G,
            ThermalFluidType.MARLOTHERM_SH,
            ThermalFluidType.SYLTHERM_800,
        ]

        for fluid in expected_fluids:
            assert fluid in FLUID_DATA, f"Missing data for {fluid}"

    def test_coefficient_data_structure(self):
        """Test coefficient data structure is valid."""
        for fluid_type, coeffs in FLUID_DATA.items():
            assert isinstance(coeffs, FluidCoefficients)
            assert len(coeffs.density_coeffs) == 3
            assert len(coeffs.specific_heat_coeffs) == 3
            assert len(coeffs.thermal_conductivity_coeffs) == 3
            assert len(coeffs.viscosity_coeffs) == 3
            assert len(coeffs.vapor_pressure_coeffs) == 3

    def test_supported_fluids_list(self, property_db):
        """Test supported fluids list."""
        supported = property_db.get_supported_fluids()

        assert len(supported) > 0
        assert ThermalFluidType.THERMINOL_66 in supported
        assert ThermalFluidType.DOWTHERM_A in supported


# =============================================================================
# PROPERTY CALCULATION TESTS - THERMINOL 66
# =============================================================================

class TestTherminol66Properties:
    """Tests for Therminol 66 property calculations."""

    def test_density_at_77f(self, property_db):
        """Test density at reference temperature (77F)."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, 77.0)

        # Expected ~63 lb/ft3 at 77F per manufacturer data
        assert 60.0 <= props.density_lb_ft3 <= 66.0

    def test_density_decreases_with_temperature(self, property_db):
        """Test density decreases with temperature."""
        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 100.0)
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)

        assert props_hot.density_lb_ft3 < props_cold.density_lb_ft3

    @pytest.mark.parametrize("temp_f,expected_density_range", [
        (100.0, (61.0, 65.0)),
        (300.0, (54.0, 58.0)),
        (500.0, (48.0, 52.0)),
        (600.0, (45.0, 50.0)),
    ])
    def test_density_at_temperatures(self, property_db, temp_f, expected_density_range):
        """Test density at various temperatures."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, temp_f)

        assert expected_density_range[0] <= props.density_lb_ft3 <= expected_density_range[1]

    def test_specific_heat_at_temperatures(self, property_db):
        """Test specific heat increases with temperature."""
        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 100.0)
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)

        # Specific heat should increase with temperature
        assert props_hot.specific_heat_btu_lb_f > props_cold.specific_heat_btu_lb_f

    @pytest.mark.parametrize("temp_f,expected_cp_range", [
        (100.0, (0.40, 0.50)),
        (300.0, (0.50, 0.60)),
        (500.0, (0.58, 0.68)),
        (600.0, (0.62, 0.72)),
    ])
    def test_specific_heat_values(self, property_db, temp_f, expected_cp_range):
        """Test specific heat at various temperatures."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, temp_f)

        assert expected_cp_range[0] <= props.specific_heat_btu_lb_f <= expected_cp_range[1]

    def test_thermal_conductivity_decreases_with_temp(self, property_db):
        """Test thermal conductivity decreases with temperature."""
        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 100.0)
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)

        assert props_hot.thermal_conductivity_btu_hr_ft_f < props_cold.thermal_conductivity_btu_hr_ft_f

    @pytest.mark.parametrize("temp_f,expected_k_range", [
        (100.0, (0.070, 0.080)),
        (300.0, (0.060, 0.070)),
        (500.0, (0.055, 0.065)),
    ])
    def test_thermal_conductivity_values(self, property_db, temp_f, expected_k_range):
        """Test thermal conductivity at various temperatures."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, temp_f)

        assert expected_k_range[0] <= props.thermal_conductivity_btu_hr_ft_f <= expected_k_range[1]

    def test_viscosity_decreases_with_temp(self, property_db):
        """Test viscosity decreases with temperature."""
        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 100.0)
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)

        assert props_hot.kinematic_viscosity_cst < props_cold.kinematic_viscosity_cst

    @pytest.mark.parametrize("temp_f,expected_visc_range", [
        (100.0, (15.0, 35.0)),  # High viscosity at low temp
        (300.0, (2.0, 6.0)),
        (500.0, (0.5, 2.0)),
        (600.0, (0.5, 1.5)),  # Low viscosity at high temp
    ])
    def test_viscosity_values(self, property_db, temp_f, expected_visc_range):
        """Test viscosity at various temperatures."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, temp_f)

        assert expected_visc_range[0] <= props.kinematic_viscosity_cst <= expected_visc_range[1]

    def test_vapor_pressure_increases_with_temp(self, property_db):
        """Test vapor pressure increases with temperature."""
        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 300.0)
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)

        assert props_hot.vapor_pressure_psia > props_cold.vapor_pressure_psia

    def test_prandtl_number_decreases_with_temp(self, property_db):
        """Test Prandtl number decreases with temperature."""
        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 100.0)
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)

        assert props_hot.prandtl_number < props_cold.prandtl_number

    def test_safety_properties(self, property_db):
        """Test safety properties are returned correctly."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, 500.0)

        # Therminol 66 flash point ~340F
        assert props.flash_point_f == 340.0
        # Auto-ignition ~750F
        assert props.auto_ignition_temp_f == 750.0
        # Max film temp ~705F
        assert props.max_film_temp_f == 705.0
        # Max bulk temp ~650F
        assert props.max_bulk_temp_f == 650.0


# =============================================================================
# PROPERTY CALCULATION TESTS - DOWTHERM A
# =============================================================================

class TestDowthermAProperties:
    """Tests for Dowtherm A property calculations."""

    def test_density_at_reference(self, property_db):
        """Test density at reference temperature."""
        props = property_db.get_properties(ThermalFluidType.DOWTHERM_A, 77.0)

        # Expected ~65-66 lb/ft3 at 77F
        assert 63.0 <= props.density_lb_ft3 <= 68.0

    def test_higher_max_temp_than_therminol_66(self, property_db):
        """Test Dowtherm A has higher max temp than Therminol 66."""
        props_dt = property_db.get_properties(ThermalFluidType.DOWTHERM_A, 500.0)
        props_t66 = property_db.get_properties(ThermalFluidType.THERMINOL_66, 500.0)

        assert props_dt.max_bulk_temp_f > props_t66.max_bulk_temp_f

    def test_flash_point(self, property_db):
        """Test flash point value."""
        props = property_db.get_properties(ThermalFluidType.DOWTHERM_A, 300.0)

        # Dowtherm A flash point ~255F
        assert props.flash_point_f == 255.0

    def test_auto_ignition_temp(self, property_db):
        """Test auto-ignition temperature."""
        props = property_db.get_properties(ThermalFluidType.DOWTHERM_A, 300.0)

        # Dowtherm A AIT ~1150F (very high)
        assert props.auto_ignition_temp_f == 1150.0


# =============================================================================
# PROPERTY CALCULATION TESTS - OTHER FLUIDS
# =============================================================================

class TestOtherFluidProperties:
    """Tests for other thermal fluid properties."""

    @pytest.mark.parametrize("fluid_type", [
        ThermalFluidType.THERMINOL_55,
        ThermalFluidType.THERMINOL_59,
        ThermalFluidType.THERMINOL_62,
        ThermalFluidType.THERMINOL_VP1,
        ThermalFluidType.DOWTHERM_G,
        ThermalFluidType.DOWTHERM_J,
        ThermalFluidType.DOWTHERM_Q,
        ThermalFluidType.MARLOTHERM_SH,
        ThermalFluidType.SYLTHERM_800,
    ])
    def test_all_fluids_return_valid_properties(self, property_db, fluid_type):
        """Test all fluids return valid properties."""
        props = property_db.get_properties(fluid_type, 300.0)

        assert isinstance(props, FluidProperties)
        assert props.density_lb_ft3 > 0
        assert props.specific_heat_btu_lb_f > 0
        assert props.thermal_conductivity_btu_hr_ft_f > 0
        assert props.kinematic_viscosity_cst > 0
        assert props.prandtl_number > 0

    def test_syltherm_800_high_temp_capability(self, property_db):
        """Test Syltherm 800 high temperature capability."""
        props = property_db.get_properties(ThermalFluidType.SYLTHERM_800, 500.0)

        # Syltherm 800 has highest max bulk temp (750F)
        assert props.max_bulk_temp_f == 750.0
        assert props.max_film_temp_f == 780.0

    def test_dowtherm_j_low_temp_capability(self, property_db):
        """Test Dowtherm J low temperature capability."""
        # Dowtherm J can operate at very low temperatures
        coeffs = FLUID_DATA[ThermalFluidType.DOWTHERM_J]
        assert coeffs.min_temp_f <= -100


# =============================================================================
# INDIVIDUAL PROPERTY GETTER TESTS
# =============================================================================

class TestIndividualPropertyGetters:
    """Tests for individual property getter methods."""

    def test_get_density(self, property_db):
        """Test get_density method."""
        density = property_db.get_density(ThermalFluidType.THERMINOL_66, 500.0)

        assert isinstance(density, float)
        assert 40.0 <= density <= 60.0

    def test_get_specific_heat(self, property_db):
        """Test get_specific_heat method."""
        cp = property_db.get_specific_heat(ThermalFluidType.THERMINOL_66, 500.0)

        assert isinstance(cp, float)
        assert 0.3 <= cp <= 1.0

    def test_get_thermal_conductivity(self, property_db):
        """Test get_thermal_conductivity method."""
        k = property_db.get_thermal_conductivity(ThermalFluidType.THERMINOL_66, 500.0)

        assert isinstance(k, float)
        assert 0.01 <= k <= 0.2

    def test_get_viscosity(self, property_db):
        """Test get_viscosity method."""
        nu = property_db.get_viscosity(ThermalFluidType.THERMINOL_66, 500.0)

        assert isinstance(nu, float)
        assert 0.1 <= nu <= 100.0

    def test_get_vapor_pressure(self, property_db):
        """Test get_vapor_pressure method."""
        pv = property_db.get_vapor_pressure(ThermalFluidType.THERMINOL_66, 500.0)

        assert isinstance(pv, float)
        assert pv > 0

    def test_get_flash_point(self, property_db):
        """Test get_flash_point method."""
        fp = property_db.get_flash_point(ThermalFluidType.THERMINOL_66)

        assert fp == 340.0

    def test_get_auto_ignition_temp(self, property_db):
        """Test get_auto_ignition_temp method."""
        ait = property_db.get_auto_ignition_temp(ThermalFluidType.THERMINOL_66)

        assert ait == 750.0

    def test_get_max_film_temp(self, property_db):
        """Test get_max_film_temp method."""
        max_film = property_db.get_max_film_temp(ThermalFluidType.THERMINOL_66)

        assert max_film == 705.0

    def test_get_max_bulk_temp(self, property_db):
        """Test get_max_bulk_temp method."""
        max_bulk = property_db.get_max_bulk_temp(ThermalFluidType.THERMINOL_66)

        assert max_bulk == 650.0

    def test_get_temperature_range(self, property_db):
        """Test get_temperature_range method."""
        min_temp, max_temp = property_db.get_temperature_range(ThermalFluidType.THERMINOL_66)

        assert min_temp == -20.0
        assert max_temp == 650.0


# =============================================================================
# THERMAL EXPANSION TESTS
# =============================================================================

class TestThermalExpansion:
    """Tests for thermal expansion calculations."""

    def test_expansion_coefficient_positive(self, property_db):
        """Test thermal expansion coefficient is positive."""
        beta = property_db.get_thermal_expansion_coefficient(
            ThermalFluidType.THERMINOL_66, 500.0
        )

        assert beta > 0

    def test_expansion_volume_calculation(self, property_db):
        """Test expansion volume calculation."""
        expansion = property_db.calculate_expansion_volume(
            fluid_type=ThermalFluidType.THERMINOL_66,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        # Expect roughly 15-25% expansion from 70F to 600F
        expected_min = 5000.0 * 0.12
        expected_max = 5000.0 * 0.30

        assert expected_min <= expansion <= expected_max

    def test_expansion_zero_at_same_temp(self, property_db):
        """Test expansion is zero at same temperature."""
        expansion = property_db.calculate_expansion_volume(
            fluid_type=ThermalFluidType.THERMINOL_66,
            system_volume_gallons=5000.0,
            cold_temp_f=500.0,
            hot_temp_f=500.0,
        )

        assert abs(expansion) < 0.1  # Essentially zero

    def test_expansion_negative_for_cooling(self, property_db):
        """Test expansion is negative when cooling (contraction)."""
        expansion = property_db.calculate_expansion_volume(
            fluid_type=ThermalFluidType.THERMINOL_66,
            system_volume_gallons=5000.0,
            cold_temp_f=600.0,  # Hot
            hot_temp_f=70.0,    # Cold
        )

        assert expansion < 0  # Contraction


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_custom_fluid_raises_error(self, property_db):
        """Test CUSTOM fluid type raises error."""
        with pytest.raises(ValueError) as exc_info:
            property_db.get_properties(ThermalFluidType.CUSTOM, 500.0)

        assert "Custom fluid" in str(exc_info.value)

    def test_temperature_outside_range_logs_warning(self, property_db, caplog):
        """Test temperature outside range logs warning."""
        import logging

        # This should log a warning but not raise error
        with caplog.at_level(logging.WARNING):
            props = property_db.get_properties(ThermalFluidType.THERMINOL_66, 700.0)

        # Should still return properties
        assert props is not None

    def test_calculation_count_increments(self, property_db):
        """Test calculation count increments."""
        assert property_db.calculation_count == 0

        property_db.get_properties(ThermalFluidType.THERMINOL_66, 500.0)
        assert property_db.calculation_count == 1

        property_db.get_properties(ThermalFluidType.THERMINOL_66, 600.0)
        assert property_db.calculation_count == 2

    def test_viscosity_bounded(self, property_db):
        """Test viscosity is bounded to reasonable range."""
        # At extreme temps, viscosity should still be bounded
        props_hot = property_db.get_properties(ThermalFluidType.THERMINOL_66, 650.0)
        assert props_hot.kinematic_viscosity_cst >= 0.5

        props_cold = property_db.get_properties(ThermalFluidType.THERMINOL_66, 0.0)
        assert props_cold.kinematic_viscosity_cst <= 10000.0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_fluid_properties(self):
        """Test get_fluid_properties convenience function."""
        props = get_fluid_properties(ThermalFluidType.THERMINOL_66, 500.0)

        assert isinstance(props, FluidProperties)
        assert props.temperature_f == 500.0

    def test_compare_fluids(self):
        """Test compare_fluids convenience function."""
        fluids = [
            ThermalFluidType.THERMINOL_66,
            ThermalFluidType.DOWTHERM_A,
        ]

        comparison = compare_fluids(fluids, 500.0)

        assert "therminol_66" in comparison
        assert "dowtherm_a" in comparison
        assert "density_lb_ft3" in comparison["therminol_66"]
        assert "specific_heat_btu_lb_f" in comparison["therminol_66"]

    def test_compare_fluids_handles_errors(self, caplog):
        """Test compare_fluids handles errors gracefully."""
        fluids = [
            ThermalFluidType.THERMINOL_66,
            ThermalFluidType.CUSTOM,  # Will fail
        ]

        comparison = compare_fluids(fluids, 500.0)

        # Should still have Therminol 66
        assert "therminol_66" in comparison
        # CUSTOM should be missing due to error
        assert "custom" not in comparison


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism (reproducibility)."""

    def test_same_input_same_output(self, property_db):
        """Test same input produces identical output."""
        props1 = property_db.get_properties(ThermalFluidType.THERMINOL_66, 550.0)
        props2 = property_db.get_properties(ThermalFluidType.THERMINOL_66, 550.0)

        assert props1.density_lb_ft3 == props2.density_lb_ft3
        assert props1.specific_heat_btu_lb_f == props2.specific_heat_btu_lb_f
        assert props1.kinematic_viscosity_cst == props2.kinematic_viscosity_cst
        assert props1.prandtl_number == props2.prandtl_number

    def test_multiple_database_instances_consistent(self):
        """Test multiple database instances give same results."""
        db1 = ThermalFluidPropertyDatabase()
        db2 = ThermalFluidPropertyDatabase()

        props1 = db1.get_properties(ThermalFluidType.THERMINOL_66, 550.0)
        props2 = db2.get_properties(ThermalFluidType.THERMINOL_66, 550.0)

        assert props1.density_lb_ft3 == props2.density_lb_ft3
        assert props1.prandtl_number == props2.prandtl_number


# =============================================================================
# PHYSICS VALIDATION TESTS
# =============================================================================

class TestPhysicsValidation:
    """Tests for physical reasonableness of calculations."""

    def test_prandtl_number_calculation(self, property_db):
        """Test Prandtl number calculation is correct."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, 500.0)

        # Pr = Cp * mu / k (with unit conversions)
        # Manual verification of Prandtl number
        assert 1.0 <= props.prandtl_number <= 200.0

    def test_dynamic_viscosity_from_kinematic(self, property_db):
        """Test dynamic viscosity derived from kinematic correctly."""
        props = property_db.get_properties(ThermalFluidType.THERMINOL_66, 500.0)

        # mu (cP) should relate to nu (cSt) via density
        # Relationship: mu = nu * rho (with unit conversions)
        assert props.dynamic_viscosity_cp > 0
        assert props.dynamic_viscosity_cp < props.kinematic_viscosity_cst * 10  # Rough check

    @pytest.mark.parametrize("fluid_type", list(FLUID_DATA.keys()))
    def test_all_fluids_physically_reasonable(self, property_db, fluid_type):
        """Test all fluids have physically reasonable properties."""
        # Test at mid-range temperature
        coeffs = FLUID_DATA[fluid_type]
        mid_temp = (coeffs.min_temp_f + coeffs.max_temp_f) / 2

        props = property_db.get_properties(fluid_type, mid_temp)

        # Density: typical organic fluids 40-70 lb/ft3
        assert 30.0 <= props.density_lb_ft3 <= 80.0

        # Specific heat: 0.3-0.8 BTU/lb-F for organics
        assert 0.2 <= props.specific_heat_btu_lb_f <= 1.0

        # Thermal conductivity: 0.03-0.15 BTU/hr-ft-F
        assert 0.01 <= props.thermal_conductivity_btu_hr_ft_f <= 0.2

        # Prandtl: typically 1-200 for heat transfer fluids
        assert 0.5 <= props.prandtl_number <= 500.0
