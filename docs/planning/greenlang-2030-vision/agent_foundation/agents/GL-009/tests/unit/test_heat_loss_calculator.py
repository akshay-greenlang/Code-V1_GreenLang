"""Unit tests for Heat Loss Calculator.

Tests radiation, convection, conduction, flue gas, and unburned fuel losses.

Target Coverage: 92%+
Test Count: 22+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.heat_loss_calculator import (
    HeatLossCalculator,
    HeatLossResult,
    RadiationLoss,
    ConvectionLoss,
    ConductionLoss,
    FlueGasLoss,
    UnburnedFuelLoss,
    SurfaceGeometry,
    InsulationLayer,
    FlueGasComposition,
    SurfaceOrientation,
    InsulationMaterial,
    ConvectionType,
    STEFAN_BOLTZMANN
)


class TestHeatLossCalculator:
    """Test suite for HeatLossCalculator."""

    def test_initialization(self):
        """Test calculator initializes correctly."""
        calculator = HeatLossCalculator(precision=6)

        assert calculator.precision == 6
        assert calculator.VERSION == "1.0.0"

    def test_radiation_loss_basic(self, sample_surface_geometry):
        """Test basic radiation heat loss calculation."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_radiation_loss(
            surface_temperature_k=373.15,  # 100C
            ambient_temperature_k=298.15,  # 25C
            geometry=sample_surface_geometry
        )

        assert isinstance(result, RadiationLoss)
        assert result.heat_loss_kw > 0
        assert result.emissivity == sample_surface_geometry.emissivity
        assert result.heat_flux_w_m2 > 0

    def test_radiation_loss_stefan_boltzmann_formula(self):
        """Test radiation follows Stefan-Boltzmann law."""
        calculator = HeatLossCalculator()

        geometry = SurfaceGeometry(
            surface_area_m2=10.0,
            emissivity=0.9,
            view_factor=1.0
        )

        T_s = 400.0  # K
        T_a = 300.0  # K

        result = calculator.calculate_radiation_loss(T_s, T_a, geometry)

        # Manual calculation
        expected_w = 0.9 * STEFAN_BOLTZMANN * 10.0 * 1.0 * (
            math.pow(T_s, 4) - math.pow(T_a, 4)
        )
        expected_kw = expected_w / 1000.0

        assert result.heat_loss_kw == pytest.approx(expected_kw, rel=1e-4)

    def test_natural_convection_loss_vertical(self, sample_surface_geometry):
        """Test natural convection for vertical surface."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_natural_convection_loss(
            surface_temperature_k=343.15,  # 70C
            ambient_temperature_k=298.15,  # 25C
            geometry=sample_surface_geometry
        )

        assert isinstance(result, ConvectionLoss)
        assert result.convection_type == ConvectionType.NATURAL
        assert result.heat_loss_kw > 0
        assert result.heat_transfer_coefficient > 0

    @pytest.mark.parametrize("orientation,expected_type", [
        (SurfaceOrientation.VERTICAL, ConvectionType.NATURAL),
        (SurfaceOrientation.HORIZONTAL_TOP, ConvectionType.NATURAL),
        (SurfaceOrientation.HORIZONTAL_BOTTOM, ConvectionType.NATURAL),
    ])
    def test_natural_convection_various_orientations(self, orientation, expected_type):
        """Test natural convection for different surface orientations."""
        calculator = HeatLossCalculator()

        geometry = SurfaceGeometry(
            surface_area_m2=25.0,
            length_m=5.0,
            orientation=orientation
        )

        result = calculator.calculate_natural_convection_loss(
            343.15, 298.15, geometry
        )

        assert result.convection_type == expected_type
        assert result.heat_loss_kw > 0

    def test_forced_convection_loss(self, sample_surface_geometry):
        """Test forced convection heat loss."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_forced_convection_loss(
            surface_temperature_k=343.15,
            ambient_temperature_k=298.15,
            geometry=sample_surface_geometry,
            air_velocity_m_s=5.0
        )

        assert result.convection_type == ConvectionType.FORCED
        assert result.air_velocity_m_s == 5.0
        assert result.heat_loss_kw > 0

    def test_forced_convection_zero_velocity_raises_error(self, sample_surface_geometry):
        """Test forced convection rejects zero air velocity."""
        calculator = HeatLossCalculator()

        with pytest.raises(ValueError, match="Air velocity must be positive"):
            calculator.calculate_forced_convection_loss(
                343.15, 298.15, sample_surface_geometry, air_velocity_m_s=0.0
            )

    def test_conduction_loss_single_layer(self):
        """Test conduction loss through single insulation layer."""
        calculator = HeatLossCalculator()

        layers = [
            InsulationLayer(
                material=InsulationMaterial.MINERAL_WOOL,
                thickness_m=0.1,
                thermal_conductivity_w_mk=0.045
            )
        ]

        result = calculator.calculate_conduction_loss(
            hot_side_temperature_k=373.15,
            cold_side_temperature_k=298.15,
            surface_area_m2=20.0,
            insulation_layers=layers
        )

        assert isinstance(result, ConductionLoss)
        assert result.heat_loss_kw > 0
        assert result.total_resistance_k_w > 0
        assert len(result.layer_resistances) == 1

    def test_conduction_loss_multiple_layers(self, sample_insulation_layers):
        """Test conduction loss through multiple layers."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_conduction_loss(
            hot_side_temperature_k=473.15,
            cold_side_temperature_k=298.15,
            surface_area_m2=50.0,
            insulation_layers=sample_insulation_layers
        )

        assert len(result.layer_resistances) == 2
        assert result.total_resistance_k_w > 0

    def test_conduction_fourier_law(self):
        """Test conduction follows Fourier's law."""
        calculator = HeatLossCalculator()

        T_hot = 400.0
        T_cold = 300.0
        A = 10.0
        thickness = 0.1
        k = 0.05

        layers = [
            InsulationLayer(
                material=InsulationMaterial.MINERAL_WOOL,
                thickness_m=thickness,
                thermal_conductivity_w_mk=k
            )
        ]

        result = calculator.calculate_conduction_loss(T_hot, T_cold, A, layers)

        # Q = (T_hot - T_cold) / R
        # R = thickness / (k * A)
        R_expected = thickness / (k * A)
        Q_expected_kw = (T_hot - T_cold) / R_expected / 1000.0

        assert result.heat_loss_kw == pytest.approx(Q_expected_kw, rel=1e-4)

    def test_flue_gas_loss_calculation(self, sample_flue_gas_composition):
        """Test flue gas heat loss calculation."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_flue_gas_loss(
            flue_gas_temperature_k=423.15,  # 150C
            ambient_temperature_k=298.15,  # 25C
            flue_gas_flow_kg_s=5.0,
            flue_gas_composition=sample_flue_gas_composition,
            fuel_energy_input_kw=10000.0,
            fuel_hydrogen_percent=15.0
        )

        assert isinstance(result, FlueGasLoss)
        assert result.total_loss_kw > 0
        assert result.sensible_loss_kw > 0
        assert result.dry_loss_percent > 0

    def test_flue_gas_sensible_heat(self):
        """Test flue gas sensible heat calculation."""
        calculator = HeatLossCalculator()

        composition = FlueGasComposition(
            co2_percent=12.0,
            o2_percent=6.0,
            n2_percent=75.0,
            h2o_percent=7.0
        )

        result = calculator.calculate_flue_gas_loss(
            flue_gas_temperature_k=400.0,
            ambient_temperature_k=300.0,
            flue_gas_flow_kg_s=10.0,
            flue_gas_composition=composition,
            fuel_energy_input_kw=10000.0
        )

        # Q = m * cp * dT
        # Should be > 0
        assert result.sensible_loss_kw > 0

    def test_unburned_fuel_loss_carbon_in_ash(self):
        """Test unburned fuel loss from carbon in ash."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_unburned_fuel_loss(
            fuel_energy_input_kw=10000.0,
            carbon_in_ash_percent=5.0,
            ash_flow_kg_s=0.1,
            co_ppm=0.0,
            flue_gas_flow_kg_s=0.0
        )

        assert isinstance(result, UnburnedFuelLoss)
        assert result.combustible_in_ash_loss_kw > 0
        assert result.unburned_carbon_percent == 5.0

    def test_unburned_fuel_loss_co(self):
        """Test unburned fuel loss from CO in flue gas."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_unburned_fuel_loss(
            fuel_energy_input_kw=10000.0,
            carbon_in_ash_percent=0.0,
            ash_flow_kg_s=0.0,
            co_ppm=500.0,
            flue_gas_flow_kg_s=10.0
        )

        assert result.co_loss_kw > 0
        assert result.co_concentration_ppm == 500.0

    def test_calculate_total_losses_comprehensive(
        self,
        sample_surface_geometry,
        sample_insulation_layers,
        sample_flue_gas_composition
    ):
        """Test comprehensive calculation of all heat losses."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_total_losses(
            surface_temperature_k=343.15,
            ambient_temperature_k=298.15,
            surface_geometry=sample_surface_geometry,
            insulation_layers=sample_insulation_layers,
            air_velocity_m_s=None,  # Natural convection
            flue_gas_temperature_k=423.15,
            flue_gas_flow_kg_s=5.0,
            flue_gas_composition=sample_flue_gas_composition,
            fuel_energy_input_kw=10000.0,
            carbon_in_ash_percent=2.0,
            ash_flow_kg_s=0.05,
            co_ppm=100.0
        )

        assert isinstance(result, HeatLossResult)
        assert result.total_loss_kw > 0
        assert result.radiation_loss is not None
        assert result.convection_loss is not None
        assert result.conduction_loss is not None
        assert result.flue_gas_loss is not None
        assert result.unburned_fuel_loss is not None

    def test_loss_breakdown_percentages(self, sample_surface_geometry):
        """Test loss breakdown as percentage of total."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_total_losses(
            surface_temperature_k=343.15,
            ambient_temperature_k=298.15,
            surface_geometry=sample_surface_geometry
        )

        # Sum of percentages should be ~100%
        total_percent = sum(result.loss_breakdown_percent.values())
        assert total_percent == pytest.approx(100.0, rel=0.01)

    def test_temperature_validation_negative_raises_error(self, sample_surface_geometry):
        """Test negative temperature raises error."""
        calculator = HeatLossCalculator()

        with pytest.raises(ValueError, match="Temperatures must be positive"):
            calculator.calculate_radiation_loss(
                -273.15, 298.15, sample_surface_geometry
            )

    def test_temperature_validation_hot_less_than_cold_warning(
        self, sample_surface_geometry
    ):
        """Test warning when hot temperature < cold temperature."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_total_losses(
            surface_temperature_k=298.15,  # Lower
            ambient_temperature_k=343.15,  # Higher
            surface_geometry=sample_surface_geometry
        )

        assert len(result.warnings) > 0

    def test_provenance_hash_generation(self, sample_surface_geometry):
        """Test provenance hash is generated."""
        calculator = HeatLossCalculator()

        result = calculator.calculate_total_losses(
            343.15, 298.15, sample_surface_geometry
        )

        assert len(result.provenance_hash) == 64  # SHA-256


class TestSurfaceGeometryValidation:
    """Test SurfaceGeometry validation."""

    def test_surface_geometry_valid(self):
        """Test valid surface geometry creation."""
        geometry = SurfaceGeometry(
            surface_area_m2=100.0,
            length_m=10.0,
            emissivity=0.85
        )

        assert geometry.surface_area_m2 == 100.0

    def test_surface_geometry_negative_area_raises_error(self):
        """Test negative surface area raises error."""
        with pytest.raises(ValueError, match="Surface area cannot be negative"):
            SurfaceGeometry(surface_area_m2=-10.0)

    def test_surface_geometry_invalid_emissivity_raises_error(self):
        """Test invalid emissivity raises error."""
        with pytest.raises(ValueError, match="Emissivity must be 0-1"):
            SurfaceGeometry(surface_area_m2=10.0, emissivity=1.5)

    def test_surface_geometry_invalid_view_factor_raises_error(self):
        """Test invalid view factor raises error."""
        with pytest.raises(ValueError, match="View factor must be 0-1"):
            SurfaceGeometry(surface_area_m2=10.0, view_factor=2.0)
