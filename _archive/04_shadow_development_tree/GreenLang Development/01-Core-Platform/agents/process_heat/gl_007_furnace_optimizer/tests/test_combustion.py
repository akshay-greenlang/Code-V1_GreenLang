# -*- coding: utf-8 -*-
"""
GL-007 Combustion Calculator Tests
==================================

Unit tests for GL-007 combustion analysis calculator.
Tests ASME PTC 4 efficiency calculations, excess air, and emissions.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from greenlang.agents.process_heat.gl_007_furnace_optimizer.combustion import (
    CombustionCalculator,
    CombustionConstants,
    FUEL_PROPERTIES,
    create_combustion_calculator,
)
from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    CombustionAnalysis,
    CombustionStatus,
)


class TestCombustionConstants:
    """Tests for combustion engineering constants."""

    def test_standard_conditions(self):
        """Test standard condition constants."""
        assert CombustionConstants.STD_TEMP_F == 60.0
        assert CombustionConstants.STD_PRESSURE_PSIA == 14.696

    def test_air_composition(self):
        """Test air composition constants."""
        assert CombustionConstants.AIR_O2_PCT == 20.95
        assert CombustionConstants.AIR_N2_PCT == 78.09
        # Total should be ~100%
        total = (
            CombustionConstants.AIR_O2_PCT
            + CombustionConstants.AIR_N2_PCT
            + CombustionConstants.AIR_AR_PCT
            + CombustionConstants.AIR_CO2_PCT
        )
        assert abs(total - 100.0) < 0.1

    def test_molecular_weights(self):
        """Test molecular weight constants."""
        assert CombustionConstants.MW_O2 == 32.00
        assert CombustionConstants.MW_N2 == 28.01
        assert CombustionConstants.MW_CO2 == 44.01
        assert CombustionConstants.MW_H2O == 18.02

    def test_specific_heats(self):
        """Test specific heat constants."""
        assert CombustionConstants.CP_FLUE_GAS > 0
        assert CombustionConstants.CP_AIR > 0
        assert CombustionConstants.CP_WATER_VAPOR > 0


class TestFuelProperties:
    """Tests for fuel properties database."""

    def test_natural_gas_properties(self):
        """Test natural gas fuel properties."""
        props = FUEL_PROPERTIES["natural_gas"]
        assert props["hhv_btu_scf"] == 1020.0
        assert props["lhv_btu_scf"] == 920.0
        assert props["stoich_air_scf_per_scf"] == 9.52
        assert props["co2_factor_lb_mmbtu"] == 117.0

    def test_propane_properties(self):
        """Test propane fuel properties."""
        props = FUEL_PROPERTIES["propane"]
        assert props["hhv_btu_scf"] == 2516.0
        assert props["stoich_air_scf_per_scf"] == 23.81

    def test_hydrogen_properties(self):
        """Test hydrogen fuel properties."""
        props = FUEL_PROPERTIES["hydrogen"]
        assert props["h2_in_fuel_pct"] == 100.0
        assert props["c_in_fuel_pct"] == 0.0
        assert props["co2_factor_lb_mmbtu"] == 0.0  # Zero carbon

    def test_all_fuels_have_required_fields(self):
        """Test all fuels have required properties."""
        required_fields = ["hhv_btu_scf", "specific_gravity", "co2_factor_lb_mmbtu"]
        for fuel_type, props in FUEL_PROPERTIES.items():
            for field in required_fields:
                assert field in props, f"{fuel_type} missing {field}"


class TestCombustionCalculator:
    """Tests for CombustionCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator(provenance_enabled=True)

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None
        assert calculator.VERSION == "1.0.0"
        assert calculator.provenance_enabled is True

    def test_initialization_without_provenance(self):
        """Test calculator without provenance tracking."""
        calc = CombustionCalculator(provenance_enabled=False)
        assert calc._provenance_tracker is None


class TestExcessAirCalculation:
    """Tests for excess air calculations."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_excess_air_from_o2_typical(self, calculator):
        """Test typical excess air calculation."""
        # At 3% O2: EA = 3 / (21-3) * 100 = 16.67%
        excess_air = calculator.calculate_excess_air_from_o2(3.0)
        assert abs(excess_air - 16.67) < 0.5

    def test_excess_air_from_o2_low(self, calculator):
        """Test low O2 excess air calculation."""
        # At 1% O2: EA = 1 / (21-1) * 100 = 5%
        excess_air = calculator.calculate_excess_air_from_o2(1.0)
        assert abs(excess_air - 5.0) < 0.5

    def test_excess_air_from_o2_high(self, calculator):
        """Test high O2 excess air calculation."""
        # At 10% O2: EA = 10 / (21-10) * 100 = 90.9%
        excess_air = calculator.calculate_excess_air_from_o2(10.0)
        assert abs(excess_air - 90.9) < 1.0

    def test_excess_air_from_o2_invalid_high(self, calculator):
        """Test invalid high O2 value."""
        with pytest.raises(ValueError):
            calculator.calculate_excess_air_from_o2(21.0)  # At or above 21% is invalid

    def test_excess_air_from_o2_invalid_negative(self, calculator):
        """Test invalid negative O2 value."""
        with pytest.raises(ValueError):
            calculator.calculate_excess_air_from_o2(-1.0)

    def test_excess_air_from_co2(self, calculator):
        """Test excess air from CO2 measurement."""
        # At 9% CO2 (theoretical ~11.7%): EA = (11.7/9 - 1) * 100 = 30%
        excess_air = calculator.calculate_excess_air_from_co2(9.0, theoretical_co2_pct=11.7)
        assert abs(excess_air - 30.0) < 1.0

    def test_excess_air_from_co2_invalid(self, calculator):
        """Test invalid CO2 value."""
        with pytest.raises(ValueError):
            calculator.calculate_excess_air_from_co2(0.0)


class TestStoichiometricAir:
    """Tests for stoichiometric air calculations."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_natural_gas_stoich_air(self, calculator):
        """Test natural gas stoichiometric air."""
        stoich = calculator.calculate_stoichiometric_air(fuel_type="natural_gas")
        assert 9.0 <= stoich <= 10.5

    def test_propane_stoich_air(self, calculator):
        """Test propane stoichiometric air."""
        stoich = calculator.calculate_stoichiometric_air(fuel_type="propane")
        assert 20.0 <= stoich <= 25.0

    def test_hydrogen_stoich_air(self, calculator):
        """Test hydrogen stoichiometric air."""
        stoich = calculator.calculate_stoichiometric_air(fuel_type="hydrogen")
        assert 2.0 <= stoich <= 3.0

    def test_stoich_air_with_hhv_adjustment(self, calculator):
        """Test stoichiometric air with HHV adjustment."""
        # Higher HHV should require more air
        stoich_std = calculator.calculate_stoichiometric_air(
            fuel_type="natural_gas",
            fuel_hhv_btu_scf=1020.0,
        )
        stoich_high = calculator.calculate_stoichiometric_air(
            fuel_type="natural_gas",
            fuel_hhv_btu_scf=1100.0,
        )
        assert stoich_high > stoich_std


class TestFlueGasComposition:
    """Tests for flue gas composition calculations."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_flue_gas_composition_typical(self, calculator):
        """Test typical flue gas composition."""
        comp = calculator.calculate_flue_gas_composition(
            fuel_type="natural_gas",
            excess_air_pct=15.0,
        )

        assert "o2_pct" in comp
        assert "co2_pct" in comp
        assert "n2_pct" in comp
        assert "h2o_pct" in comp

    def test_flue_gas_o2_at_excess_air(self, calculator):
        """Test O2 in flue gas correlates with excess air."""
        comp_low = calculator.calculate_flue_gas_composition(excess_air_pct=10.0)
        comp_high = calculator.calculate_flue_gas_composition(excess_air_pct=30.0)

        # Higher excess air = higher O2
        assert comp_high["o2_pct"] > comp_low["o2_pct"]

    def test_flue_gas_co2_dilution(self, calculator):
        """Test CO2 dilution with excess air."""
        comp_low = calculator.calculate_flue_gas_composition(excess_air_pct=10.0)
        comp_high = calculator.calculate_flue_gas_composition(excess_air_pct=30.0)

        # Higher excess air dilutes CO2
        assert comp_high["co2_pct"] < comp_low["co2_pct"]


class TestHeatLosses:
    """Tests for heat loss calculations per ASME PTC 4."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_heat_losses_typical(self, calculator):
        """Test typical heat loss calculation."""
        losses = calculator.calculate_heat_losses(
            flue_gas_temp_f=450.0,
            ambient_temp_f=77.0,
            excess_air_pct=15.0,
            fuel_type="natural_gas",
        )

        assert "dry_gas_loss" in losses
        assert "moisture_loss" in losses
        assert "radiation_loss" in losses
        assert "unburned_fuel_loss" in losses

    def test_dry_gas_loss_increases_with_temp(self, calculator):
        """Test dry gas loss increases with flue gas temperature."""
        losses_low = calculator.calculate_heat_losses(
            flue_gas_temp_f=350.0,
            ambient_temp_f=77.0,
            excess_air_pct=15.0,
        )
        losses_high = calculator.calculate_heat_losses(
            flue_gas_temp_f=500.0,
            ambient_temp_f=77.0,
            excess_air_pct=15.0,
        )

        assert losses_high["dry_gas_loss"] > losses_low["dry_gas_loss"]

    def test_dry_gas_loss_increases_with_excess_air(self, calculator):
        """Test dry gas loss increases with excess air."""
        losses_low = calculator.calculate_heat_losses(
            flue_gas_temp_f=450.0,
            ambient_temp_f=77.0,
            excess_air_pct=10.0,
        )
        losses_high = calculator.calculate_heat_losses(
            flue_gas_temp_f=450.0,
            ambient_temp_f=77.0,
            excess_air_pct=30.0,
        )

        assert losses_high["dry_gas_loss"] > losses_low["dry_gas_loss"]

    def test_unburned_fuel_loss_with_co(self, calculator):
        """Test unburned fuel loss when CO is present."""
        losses = calculator.calculate_heat_losses(
            flue_gas_temp_f=450.0,
            ambient_temp_f=77.0,
            excess_air_pct=15.0,
            flue_gas_co_ppm=100.0,
        )

        assert losses["unburned_fuel_loss"] > 0


class TestAdiabaticFlameTemperature:
    """Tests for adiabatic flame temperature calculations."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_natural_gas_flame_temp(self, calculator):
        """Test natural gas flame temperature."""
        temp = calculator.calculate_adiabatic_flame_temp(
            fuel_type="natural_gas",
            excess_air_pct=15.0,
        )

        # Natural gas flame temp typically 3200-3600F at stoich
        assert 3000 <= temp <= 3700

    def test_flame_temp_decreases_with_excess_air(self, calculator):
        """Test flame temp decreases with excess air."""
        temp_low = calculator.calculate_adiabatic_flame_temp(excess_air_pct=10.0)
        temp_high = calculator.calculate_adiabatic_flame_temp(excess_air_pct=30.0)

        assert temp_high < temp_low

    def test_flame_temp_increases_with_preheat(self, calculator):
        """Test flame temp increases with air preheat."""
        temp_cold = calculator.calculate_adiabatic_flame_temp(air_preheat_temp_f=77.0)
        temp_hot = calculator.calculate_adiabatic_flame_temp(air_preheat_temp_f=500.0)

        assert temp_hot > temp_cold


class TestOptimalExcessAir:
    """Tests for optimal excess air recommendations."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_natural_gas_optimal_values(self, calculator):
        """Test natural gas optimal excess air values."""
        optimal = calculator.calculate_optimal_excess_air(fuel_type="natural_gas")

        assert "min_safe_excess_air" in optimal
        assert "optimal_excess_air" in optimal
        assert "optimal_o2" in optimal

        # Typical optimal is 10-20% for natural gas
        assert 10 <= optimal["optimal_excess_air"] <= 20

    def test_fuel_oil_optimal_values(self, calculator):
        """Test fuel oil optimal excess air values."""
        optimal = calculator.calculate_optimal_excess_air(fuel_type="fuel_oil_2")

        # Fuel oil requires more excess air than gas
        assert optimal["optimal_excess_air"] >= 15


class TestCombustionAnalysis:
    """Tests for complete combustion analysis."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_analyze_combustion_basic(self, calculator):
        """Test basic combustion analysis."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=3.0,
        )

        assert isinstance(result, CombustionAnalysis)
        assert result.analysis_id is not None
        assert result.thermal_efficiency_pct > 0
        assert result.excess_air_pct > 0

    def test_analyze_combustion_heat_input(self, calculator):
        """Test heat input calculation."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=3.0,
        )

        # Heat input = 5000 * 1020 / 1e6 = 5.1 MMBtu/hr
        assert abs(result.heat_input_mmbtu_hr - 5.1) < 0.1

    def test_analyze_combustion_efficiency_range(self, calculator):
        """Test efficiency is in valid range."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=3.0,
        )

        # Typical furnace efficiency 75-95%
        assert 70 <= result.thermal_efficiency_pct <= 98
        assert 70 <= result.combustion_efficiency_pct <= 100

    def test_analyze_combustion_status_optimal(self, calculator):
        """Test optimal combustion status."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=400.0,
            flue_gas_o2_pct=3.0,
            flue_gas_co_ppm=50.0,
        )

        # With good O2 and low CO, should be optimal
        assert result.combustion_status in [CombustionStatus.OPTIMAL, CombustionStatus.LEAN]

    def test_analyze_combustion_status_lean(self, calculator):
        """Test lean combustion status."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=8.0,  # High O2 = lean
        )

        assert result.combustion_status == CombustionStatus.LEAN

    def test_analyze_combustion_provenance_hash(self, calculator):
        """Test provenance hash is generated."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=3.0,
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_analyze_combustion_co2_emissions(self, calculator):
        """Test CO2 emission calculation."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=450.0,
            flue_gas_o2_pct=3.0,
            fuel_type="natural_gas",
        )

        # Natural gas CO2 factor ~117 lb/MMBtu
        assert 100 <= result.co2_lb_mmbtu <= 130

    def test_analyze_combustion_recommendations(self, calculator):
        """Test recommendations are generated."""
        result = calculator.analyze_combustion(
            fuel_flow_scfh=5000.0,
            fuel_hhv_btu_scf=1020.0,
            flue_gas_temp_f=600.0,  # High temp
            flue_gas_o2_pct=7.0,    # High O2
            flue_gas_co_ppm=150.0,  # High CO
        )

        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0


class TestInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_negative_fuel_flow(self, calculator):
        """Test negative fuel flow is rejected."""
        with pytest.raises(ValueError):
            calculator.analyze_combustion(
                fuel_flow_scfh=-100.0,
                fuel_hhv_btu_scf=1020.0,
                flue_gas_temp_f=450.0,
                flue_gas_o2_pct=3.0,
            )

    def test_invalid_hhv(self, calculator):
        """Test invalid HHV is rejected."""
        with pytest.raises(ValueError):
            calculator.analyze_combustion(
                fuel_flow_scfh=5000.0,
                fuel_hhv_btu_scf=0.0,  # Invalid
                flue_gas_temp_f=450.0,
                flue_gas_o2_pct=3.0,
            )

    def test_invalid_flue_gas_temp(self, calculator):
        """Test invalid flue gas temp is rejected."""
        with pytest.raises(ValueError):
            calculator.analyze_combustion(
                fuel_flow_scfh=5000.0,
                fuel_hhv_btu_scf=1020.0,
                flue_gas_temp_f=50.0,  # Too low
                flue_gas_o2_pct=3.0,
            )

    def test_invalid_o2(self, calculator):
        """Test invalid O2 is rejected."""
        with pytest.raises(ValueError):
            calculator.analyze_combustion(
                fuel_flow_scfh=5000.0,
                fuel_hhv_btu_scf=1020.0,
                flue_gas_temp_f=450.0,
                flue_gas_o2_pct=22.0,  # Invalid
            )


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_combustion_calculator_default(self):
        """Test default factory creation."""
        calc = create_combustion_calculator()
        assert calc is not None
        assert calc.provenance_enabled is True

    def test_create_combustion_calculator_custom(self):
        """Test custom factory creation."""
        calc = create_combustion_calculator(
            provenance_enabled=False,
            precision=2,
        )
        assert calc.provenance_enabled is False
        assert calc.precision == 2


class TestDeterminism:
    """Tests for deterministic calculation behavior."""

    @pytest.fixture
    def calculator(self):
        """Create combustion calculator instance."""
        return CombustionCalculator()

    def test_excess_air_determinism(self, calculator):
        """Test excess air calculation is deterministic."""
        results = [calculator.calculate_excess_air_from_o2(3.0) for _ in range(5)]
        assert len(set(results)) == 1  # All results identical

    def test_analysis_determinism(self, calculator):
        """Test full analysis is deterministic."""
        results = []
        for _ in range(3):
            result = calculator.analyze_combustion(
                fuel_flow_scfh=5000.0,
                fuel_hhv_btu_scf=1020.0,
                flue_gas_temp_f=450.0,
                flue_gas_o2_pct=3.0,
            )
            results.append(result.thermal_efficiency_pct)

        assert len(set(results)) == 1  # All results identical
