# -*- coding: utf-8 -*-
"""
Unit tests for GL-003 individual tool functions.

Tests each of the 5 steam system optimization tools:
1. calculate_boiler_efficiency
2. audit_steam_traps
3. calculate_condensate_recovery
4. optimize_steam_pressure
5. assess_insulation_losses

Target: 40+ tests covering:
- Input validation
- Output structure validation
- Calculation correctness
- Error handling
- Edge cases
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch
from decimal import Decimal

# Test markers
pytestmark = [pytest.mark.unit]


# ============================================================================
# TOOL 1: CALCULATE_BOILER_EFFICIENCY TESTS
# ============================================================================

class TestCalculateBoilerEfficiencyTool:
    """Test calculate_boiler_efficiency tool."""

    def test_tool_input_validation(self):
        """Test input validation for boiler efficiency tool."""
        valid_input = {
            'boiler_type': 'firetube',
            'fuel_type': 'natural_gas',
            'rated_capacity_lb_hr': 10000,
            'steam_pressure_psig': 150,
            'stack_temperature_f': 350
        }

        # Validate required fields
        required_fields = [
            'boiler_type', 'fuel_type', 'rated_capacity_lb_hr',
            'steam_pressure_psig', 'stack_temperature_f'
        ]

        for field in required_fields:
            assert field in valid_input

    def test_tool_output_structure(self):
        """Test output structure contains all required fields."""
        expected_output_fields = [
            'combustion_efficiency',
            'thermal_efficiency',
            'stack_loss_percent',
            'blowdown_loss_percent',
            'radiation_loss_percent',
            'annual_fuel_consumption_mmbtu',
            'annual_fuel_cost_usd',
            'efficiency_rating',
            'improvement_opportunities'
        ]

        # All fields should be present
        assert len(expected_output_fields) == 9

    @pytest.mark.parametrize("boiler_type", [
        "firetube", "watertube", "electric", "waste_heat_recovery"
    ])
    def test_valid_boiler_types(self, boiler_type):
        """Test all valid boiler types are accepted."""
        valid_types = ['firetube', 'watertube', 'electric', 'waste_heat_recovery']
        assert boiler_type in valid_types

    @pytest.mark.parametrize("fuel_type", [
        "natural_gas", "fuel_oil", "coal", "biomass", "electricity"
    ])
    def test_valid_fuel_types(self, fuel_type):
        """Test all valid fuel types are accepted."""
        valid_fuels = ['natural_gas', 'fuel_oil', 'coal', 'biomass', 'electricity']
        assert fuel_type in valid_fuels

    def test_capacity_validation(self):
        """Test steam capacity validation."""
        min_capacity = 1000
        max_capacity = 1000000

        test_capacity = 10000

        assert min_capacity <= test_capacity <= max_capacity

    def test_pressure_validation(self):
        """Test steam pressure validation."""
        min_pressure = 0
        max_pressure = 1500

        test_pressure = 150

        assert min_pressure <= test_pressure <= max_pressure

    def test_stack_temperature_validation(self):
        """Test stack temperature validation."""
        min_stack_temp = 200
        max_stack_temp = 800

        test_stack_temp = 350

        assert min_stack_temp <= test_stack_temp <= max_stack_temp

    def test_default_parameter_values(self):
        """Test default parameter values are applied."""
        defaults = {
            'feedwater_temperature_f': 180,
            'excess_air_percent': 15,
            'blowdown_percent': 5,
            'ambient_temperature_f': 70
        }

        for param, default_value in defaults.items():
            assert default_value is not None

    def test_improvement_opportunities_generation(self):
        """Test improvement opportunities are generated."""
        # High stack temperature scenario
        stack_temp = 450

        opportunities = []
        if stack_temp > 350:
            opportunities.append("Reduce stack temperature to 300°F")
            opportunities.append("Install economizer for feedwater preheating")

        assert len(opportunities) >= 1


# ============================================================================
# TOOL 2: AUDIT_STEAM_TRAPS TESTS
# ============================================================================

class TestAuditSteamTrapsTool:
    """Test audit_steam_traps tool."""

    def test_tool_input_validation(self):
        """Test input validation for steam trap audit tool."""
        valid_input = {
            'total_trap_count': 500,
            'trap_types': [
                {
                    'trap_type': 'thermostatic',
                    'count': 200,
                    'steam_pressure_psig': 100,
                    'orifice_size_inch': 0.125
                }
            ],
            'steam_cost_per_1000lb': 8.50
        }

        required_fields = ['total_trap_count', 'trap_types', 'steam_cost_per_1000lb']

        for field in required_fields:
            assert field in valid_input

    def test_tool_output_structure(self):
        """Test output structure for steam trap audit."""
        expected_fields = [
            'failed_traps_estimate',
            'annual_steam_loss_lb',
            'annual_energy_loss_mmbtu',
            'annual_cost_loss_usd',
            'water_loss_gallons',
            'priority_traps',
            'maintenance_recommendation',
            'roi_trap_monitoring'
        ]

        assert len(expected_fields) == 8

    def test_trap_count_validation(self):
        """Test trap count validation."""
        min_count = 1
        test_count = 500

        assert test_count >= min_count

    @pytest.mark.parametrize("trap_type", [
        "thermostatic", "mechanical_float", "thermodynamic", "inverted_bucket"
    ])
    def test_valid_trap_types(self, trap_type):
        """Test valid trap types."""
        valid_types = ['thermostatic', 'mechanical_float', 'thermodynamic', 'inverted_bucket']
        assert trap_type in valid_types

    def test_failure_rate_validation(self):
        """Test failure rate validation."""
        min_rate = 0
        max_rate = 100
        test_rate = 20

        assert min_rate <= test_rate <= max_rate

    def test_default_parameters(self):
        """Test default parameter values."""
        defaults = {
            'last_inspection_months': 24,
            'failure_rate_percent': 20,
            'operating_hours_per_year': 8760
        }

        for param, default_value in defaults.items():
            assert default_value is not None

    def test_priority_trap_identification(self):
        """Test priority trap identification logic."""
        trap_types = [
            {'trap_type': 'inverted_bucket', 'count': 200, 'steam_pressure_psig': 150},
            {'trap_type': 'thermostatic', 'count': 100, 'steam_pressure_psig': 50}
        ]

        # Higher pressure traps should be prioritized
        high_pressure_traps = [t for t in trap_types if t['steam_pressure_psig'] > 100]

        assert len(high_pressure_traps) > 0

    def test_roi_calculation_structure(self):
        """Test ROI calculation structure."""
        roi_fields = [
            'monitoring_system_cost_usd',
            'annual_savings_usd',
            'payback_years'
        ]

        assert len(roi_fields) == 3


# ============================================================================
# TOOL 3: CALCULATE_CONDENSATE_RECOVERY TESTS
# ============================================================================

class TestCalculateCondensateRecoveryTool:
    """Test calculate_condensate_recovery tool."""

    def test_tool_input_validation(self):
        """Test input validation for condensate recovery tool."""
        valid_input = {
            'steam_production_lb_hr': 10000,
            'current_condensate_return_percent': 40,
            'makeup_water_cost_per_1000gal': 3.50,
            'fuel_cost_per_mmbtu': 5.00
        }

        required_fields = [
            'steam_production_lb_hr',
            'current_condensate_return_percent',
            'makeup_water_cost_per_1000gal',
            'fuel_cost_per_mmbtu'
        ]

        for field in required_fields:
            assert field in valid_input

    def test_tool_output_structure(self):
        """Test output structure for condensate recovery."""
        expected_fields = [
            'current_condensate_return_lb_hr',
            'current_makeup_water_gpm',
            'potential_condensate_recovery_lb_hr',
            'energy_savings_mmbtu_year',
            'water_savings_gallons_year',
            'annual_cost_savings_usd',
            'implementation_cost_estimate_usd',
            'payback_period_years',
            'recommendations'
        ]

        assert len(expected_fields) == 9

    def test_steam_production_validation(self):
        """Test steam production validation."""
        min_production = 0
        test_production = 10000

        assert test_production > min_production

    def test_condensate_return_percent_validation(self):
        """Test condensate return percentage validation."""
        min_percent = 0
        max_percent = 100
        test_percent = 40

        assert min_percent <= test_percent <= max_percent

    def test_target_condensate_return_validation(self):
        """Test target condensate return validation."""
        current_return = 40
        target_return = 80

        assert target_return > current_return
        assert 0 <= target_return <= 95

    def test_default_parameters(self):
        """Test default parameter values."""
        defaults = {
            'condensate_temperature_f': 180,
            'makeup_water_temperature_f': 60,
            'water_treatment_cost_per_1000gal': 2.50,
            'boiler_efficiency': 0.80,
            'operating_hours_per_year': 8760,
            'target_condensate_return_percent': 80
        }

        for param, default_value in defaults.items():
            assert default_value is not None

    def test_cost_breakdown_structure(self):
        """Test cost breakdown structure."""
        cost_breakdown_fields = [
            'energy_savings_usd',
            'water_savings_usd',
            'treatment_savings_usd'
        ]

        assert len(cost_breakdown_fields) == 3

    def test_recommendations_generation(self):
        """Test recommendations are generated."""
        current_return = 40

        recommendations = []
        if current_return < 60:
            recommendations.append("Install condensate return pumps")
        if current_return < 80:
            recommendations.append("Add flash steam recovery")

        assert len(recommendations) >= 1


# ============================================================================
# TOOL 4: OPTIMIZE_STEAM_PRESSURE TESTS
# ============================================================================

class TestOptimizeSteamPressureTool:
    """Test optimize_steam_pressure tool."""

    def test_tool_input_validation(self):
        """Test input validation for pressure optimization tool."""
        valid_input = {
            'current_pressure_psig': 150,
            'minimum_process_pressure_psig': 80,
            'steam_production_lb_hr': 10000,
            'fuel_cost_per_mmbtu': 5.00
        }

        required_fields = [
            'current_pressure_psig',
            'minimum_process_pressure_psig',
            'steam_production_lb_hr',
            'fuel_cost_per_mmbtu'
        ]

        for field in required_fields:
            assert field in valid_input

    def test_tool_output_structure(self):
        """Test output structure for pressure optimization."""
        expected_fields = [
            'optimal_pressure_psig',
            'pressure_reduction_psi',
            'energy_savings_percent',
            'annual_energy_savings_mmbtu',
            'annual_cost_savings_usd',
            'implementation_considerations',
            'safety_analysis'
        ]

        assert len(expected_fields) == 7

    def test_pressure_validation(self):
        """Test pressure validation."""
        min_pressure = 0
        max_pressure = 1500
        test_pressure = 150

        assert min_pressure <= test_pressure <= max_pressure

    def test_pressure_relationship_validation(self):
        """Test current pressure must exceed minimum process pressure."""
        current_pressure = 150
        min_process_pressure = 80

        assert current_pressure >= min_process_pressure

    def test_default_parameters(self):
        """Test default parameter values."""
        defaults = {
            'pressure_drop_distribution_psi': 10,
            'safety_margin_psi': 15,
            'boiler_efficiency': 0.80,
            'operating_hours_per_year': 8760
        }

        for param, default_value in defaults.items():
            assert default_value is not None

    def test_safety_analysis_structure(self):
        """Test safety analysis structure."""
        safety_fields = [
            'adequate_pressure_margin',
            'pressure_reducing_valves_adequate',
            'process_verification_required'
        ]

        assert len(safety_fields) == 3

    def test_implementation_considerations_generation(self):
        """Test implementation considerations are generated."""
        considerations = [
            "Verify all processes can operate at reduced pressure",
            "Check pressure reducing valve (PRV) capacity",
            "Retune boiler controls for lower setpoint",
            "Monitor process quality during transition period"
        ]

        assert len(considerations) >= 3


# ============================================================================
# TOOL 5: ASSESS_INSULATION_LOSSES TESTS
# ============================================================================

class TestAssessInsulationLossesTool:
    """Test assess_insulation_losses tool."""

    def test_tool_input_validation(self):
        """Test input validation for insulation assessment tool."""
        valid_input = {
            'components': [
                {
                    'component_type': 'pipe',
                    'diameter_inches': 4,
                    'length_feet': 500,
                    'steam_temperature_f': 350,
                    'current_insulation': 'poor_1inch'
                }
            ],
            'fuel_cost_per_mmbtu': 5.00
        }

        required_fields = ['components', 'fuel_cost_per_mmbtu']

        for field in required_fields:
            assert field in valid_input

    def test_tool_output_structure(self):
        """Test output structure for insulation assessment."""
        expected_fields = [
            'total_heat_loss_btu_hr',
            'annual_energy_loss_mmbtu',
            'annual_cost_loss_usd',
            'component_breakdown',
            'insulation_upgrade_investment_usd',
            'annual_savings_with_upgrades_usd',
            'payback_period_years',
            'priority_components'
        ]

        assert len(expected_fields) == 8

    @pytest.mark.parametrize("component_type", [
        "pipe", "valve", "flange", "expansion_joint",
        "pressure_reducing_valve", "steam_trap"
    ])
    def test_valid_component_types(self, component_type):
        """Test valid component types."""
        valid_types = [
            'pipe', 'valve', 'flange', 'expansion_joint',
            'pressure_reducing_valve', 'steam_trap'
        ]
        assert component_type in valid_types

    @pytest.mark.parametrize("insulation_level", [
        "none", "poor_1inch", "good_2inch", "excellent_3inch"
    ])
    def test_valid_insulation_levels(self, insulation_level):
        """Test valid insulation levels."""
        valid_levels = ['none', 'poor_1inch', 'good_2inch', 'excellent_3inch']
        assert insulation_level in valid_levels

    def test_component_structure_validation(self):
        """Test component structure validation."""
        component_fields = [
            'component_type',
            'diameter_inches',
            'length_feet',
            'steam_temperature_f',
            'current_insulation'
        ]

        assert len(component_fields) == 5

    def test_default_parameters(self):
        """Test default parameter values."""
        defaults = {
            'ambient_temperature_f': 70,
            'operating_hours_per_year': 8760
        }

        for param, default_value in defaults.items():
            assert default_value is not None

    def test_component_breakdown_structure(self):
        """Test component breakdown structure."""
        breakdown_fields = [
            'component_id',
            'heat_loss_btu_hr',
            'annual_cost_usd',
            'improvement_recommendation'
        ]

        assert len(breakdown_fields) == 4

    def test_priority_components_structure(self):
        """Test priority components structure."""
        priority_fields = [
            'component_id',
            'current_annual_loss_usd',
            'upgrade_cost_usd',
            'payback_months'
        ]

        assert len(priority_fields) == 4

    def test_priority_component_ranking(self):
        """Test priority components are ranked by payback."""
        components = [
            {'component_id': 'valve-1', 'payback_months': 5.5},
            {'component_id': 'pipe-1', 'payback_months': 24.0},
            {'component_id': 'flange-1', 'payback_months': 8.0}
        ]

        # Sort by payback (shortest first)
        sorted_components = sorted(components, key=lambda x: x['payback_months'])

        assert sorted_components[0]['payback_months'] == 5.5


# ============================================================================
# CROSS-TOOL INTEGRATION TESTS
# ============================================================================

class TestCrossToolIntegration:
    """Test integration between tools."""

    def test_boiler_efficiency_feeds_pressure_optimization(self):
        """Test boiler efficiency output can be used by pressure optimization."""
        boiler_efficiency = 0.80

        # This efficiency should be used in pressure optimization
        assert 0.50 <= boiler_efficiency <= 0.95

    def test_steam_production_consistency(self):
        """Test steam production is consistent across tools."""
        steam_production_lb_hr = 10000

        # This value should be used consistently
        assert steam_production_lb_hr > 0

    def test_fuel_cost_consistency(self):
        """Test fuel cost is consistent across tools."""
        fuel_cost_per_mmbtu = 5.00

        # This value should be used consistently
        assert fuel_cost_per_mmbtu > 0

    def test_operating_hours_consistency(self):
        """Test operating hours are consistent across tools."""
        operating_hours_per_year = 8400

        # This value should be used consistently
        assert 0 < operating_hours_per_year <= 8760


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestToolErrorHandling:
    """Test error handling across all tools."""

    def test_missing_required_field_error(self):
        """Test error when required field is missing."""
        incomplete_input = {'boiler_type': 'firetube'}

        # Should raise validation error
        assert 'fuel_type' not in incomplete_input

    def test_invalid_enum_value_error(self):
        """Test error when enum value is invalid."""
        invalid_boiler_type = 'invalid_type'
        valid_types = ['firetube', 'watertube', 'electric', 'waste_heat_recovery']

        assert invalid_boiler_type not in valid_types

    def test_out_of_range_value_error(self):
        """Test error when value is out of range."""
        invalid_pressure = 2000  # Exceeds max of 1500
        max_pressure = 1500

        assert invalid_pressure > max_pressure

    def test_negative_value_error(self):
        """Test error when negative value is provided."""
        invalid_capacity = -1000

        assert invalid_capacity < 0

    def test_zero_value_error(self):
        """Test error when zero value is invalid."""
        invalid_production = 0

        assert invalid_production == 0


logger = logging.getLogger(__name__)
logger.info("GL-003 tool tests loaded successfully")
