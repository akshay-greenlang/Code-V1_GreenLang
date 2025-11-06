"""
Unit tests for ThermalStorageAgent_AI

Test Coverage:
--------------
1. Configuration Tests (5 tests)
2. Tool 1: calculate_storage_capacity (8 tests)
3. Tool 2: select_storage_technology (8 tests)
4. Tool 3: optimize_charge_discharge (7 tests)
5. Tool 4: calculate_thermal_losses (7 tests)
6. Tool 5: integrate_with_solar (7 tests)
7. Tool 6: calculate_economics (8 tests)
8. Integration Tests (4 tests)
9. Determinism Tests (3 tests)
10. Error Handling Tests (6 tests)

Total: 63+ tests
Target Coverage: 88%+

Standards:
---------
- ASHRAE Handbook HVAC Applications Ch51
- IEA ECES Annex 30 Thermal Storage
- IRENA Thermal Storage Guidelines
- ISO 9806 Solar Collector Performance
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime
import asyncio

from greenlang.agents.thermal_storage_agent_ai import ThermalStorageAgent_AI


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def agent() -> ThermalStorageAgent_AI:
    """Agent instance for testing"""
    return ThermalStorageAgent_AI(budget_usd=0.10, enable_explanations=True)


@pytest.fixture
def solar_thermal_payload() -> Dict[str, Any]:
    """Sample payload for solar thermal integration"""
    return {
        "application": "solar_thermal",
        "thermal_load_kw": 400,
        "temperature_c": 90,
        "storage_hours": 8,
        "load_profile": "continuous_24x7",
        "energy_cost_usd_per_kwh": 0.08,
        "latitude": 35.0,
        "annual_irradiance_kwh_m2": 1850,
    }


@pytest.fixture
def load_shifting_payload() -> Dict[str, Any]:
    """Sample payload for load shifting application"""
    return {
        "application": "load_shifting",
        "thermal_load_kw": 300,
        "temperature_c": 120,
        "storage_hours": 10,
        "load_profile": "daytime_only",
        "energy_cost_usd_per_kwh": 0.12,
    }


# ============================================================================
# CATEGORY 1: CONFIGURATION TESTS (5 tests)
# ============================================================================

class TestConfiguration:
    """Test agent configuration and initialization"""

    def test_default_configuration(self, agent):
        """Test default configuration values"""
        assert agent.agent_id == "industrial/thermal_storage_agent"
        assert agent.name == "ThermalStorageAgent_AI"
        assert agent.version == "1.0.0"
        assert agent.budget_usd == 0.10
        assert agent.enable_explanations is True

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.provider is not None
        assert agent._ai_call_count == 0
        assert agent._tool_call_count == 0
        assert agent._total_cost_usd == 0.0

    def test_custom_budget(self):
        """Test custom budget configuration"""
        agent = ThermalStorageAgent_AI(budget_usd=0.25)
        assert agent.budget_usd == 0.25

    def test_tool_registration(self, agent):
        """Test all 6 tools are registered"""
        assert agent.calculate_storage_capacity_tool is not None
        assert agent.select_storage_technology_tool is not None
        assert agent.optimize_charge_discharge_tool is not None
        assert agent.calculate_thermal_losses_tool is not None
        assert agent.integrate_with_solar_tool is not None
        assert agent.calculate_economics_tool is not None

    def test_tool_names(self, agent):
        """Test tool names are correct"""
        assert agent.calculate_storage_capacity_tool.name == "calculate_storage_capacity"
        assert agent.select_storage_technology_tool.name == "select_storage_technology"
        assert agent.optimize_charge_discharge_tool.name == "optimize_charge_discharge"
        assert agent.calculate_thermal_losses_tool.name == "calculate_thermal_losses"
        assert agent.integrate_with_solar_tool.name == "integrate_with_solar"
        assert agent.calculate_economics_tool.name == "calculate_economics"


# ============================================================================
# CATEGORY 2: TOOL 1 - CALCULATE STORAGE CAPACITY (8 tests)
# ============================================================================

class TestCalculateStorageCapacity:
    """Test storage capacity calculation tool"""

    def test_basic_capacity_calculation(self, agent):
        """Test basic capacity calculation for hot water storage"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=500,
            average_thermal_load_kw=400,
            storage_duration_hours=6,
            operating_temperature_c=90,
            return_temperature_c=50,
            round_trip_efficiency=0.90,
            load_profile="solar_thermal_integration"
        )

        assert "storage_capacity_kwh" in result
        assert "effective_capacity_kwh" in result
        assert "temperature_delta_k" in result
        assert "mass_storage_medium_kg" in result
        assert "volume_storage_medium_m3" in result

        # Check values are reasonable
        assert result["storage_capacity_kwh"] > 0
        assert result["temperature_delta_k"] == 40  # 90 - 50
        assert result["daily_charge_discharge_cycles"] == 1.0

    def test_high_temperature_storage(self, agent):
        """Test capacity calculation for high temperature storage"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=1000,
            average_thermal_load_kw=800,
            storage_duration_hours=8,
            operating_temperature_c=200,
            return_temperature_c=100,
            round_trip_efficiency=0.88
        )

        assert result["temperature_delta_k"] == 100
        assert result["storage_capacity_kwh"] > result["effective_capacity_kwh"]
        # Higher temp delta should result in smaller volume
        assert result["volume_storage_medium_m3"] > 0

    def test_load_shifting_profile(self, agent):
        """Test capacity calculation for load shifting"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=300,
            average_thermal_load_kw=250,
            storage_duration_hours=10,
            operating_temperature_c=80,
            return_temperature_c=40,
            load_profile="load_shifting"
        )

        assert result["daily_charge_discharge_cycles"] == 1.0
        assert "10 hours storage" in " ".join(result["sizing_notes"])

    def test_demand_response_profile(self, agent):
        """Test capacity calculation for demand response"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=200,
            average_thermal_load_kw=150,
            storage_duration_hours=4,
            operating_temperature_c=70,
            return_temperature_c=50,
            load_profile="demand_response"
        )

        assert result["daily_charge_discharge_cycles"] == 0.5  # Not every day

    def test_large_volume_recommendation(self, agent):
        """Test that large volumes get stratified tank recommendation"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=2000,
            average_thermal_load_kw=1800,
            storage_duration_hours=12,
            operating_temperature_c=95,
            return_temperature_c=55,
            round_trip_efficiency=0.92
        )

        # Large storage should recommend stratified tank
        assert result["volume_storage_medium_m3"] > 100
        assert any("stratified" in note.lower() for note in result["sizing_notes"])

    def test_low_temperature_delta_warning(self, agent):
        """Test warning for low temperature delta"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=400,
            average_thermal_load_kw=350,
            storage_duration_hours=6,
            operating_temperature_c=70,
            return_temperature_c=50,  # Only 20K delta
            round_trip_efficiency=0.90
        )

        assert result["temperature_delta_k"] < 30
        # Should have note about low delta
        assert any("low" in note.lower() for note in result["sizing_notes"])

    def test_mmbtu_conversion(self, agent):
        """Test kWh to MMBtu conversion"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=500,
            average_thermal_load_kw=400,
            storage_duration_hours=6,
            operating_temperature_c=90,
            return_temperature_c=50
        )

        # 1 kWh = 0.003412 MMBtu
        expected_mmbtu = result["storage_capacity_kwh"] * 0.003412
        assert abs(result["storage_capacity_mmbtu"] - expected_mmbtu) < 0.01

    def test_tool_call_tracking(self, agent):
        """Test that tool calls are tracked"""
        initial_count = agent._tool_call_count
        agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=100,
            average_thermal_load_kw=80,
            storage_duration_hours=4,
            operating_temperature_c=60,
            return_temperature_c=40
        )
        assert agent._tool_call_count == initial_count + 1


# ============================================================================
# CATEGORY 3: TOOL 2 - SELECT STORAGE TECHNOLOGY (8 tests)
# ============================================================================

class TestSelectStorageTechnology:
    """Test storage technology selection tool"""

    def test_hot_water_tank_selection_low_temp(self, agent):
        """Test hot water tank selection for low temperature"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=75,
            storage_duration_hours=6,
            storage_capacity_kwh=2000,
            application="solar_thermal",
            space_constraints="moderate",
            budget_constraint="medium"
        )

        assert result["recommended_technology"] == "hot_water_tank"
        assert "hot water" in result["technology_specifications"]["storage_medium"].lower()
        assert result["technology_specifications"]["round_trip_efficiency"] >= 0.90
        assert len(result["design_considerations"]) > 0

    def test_pressurized_hot_water_high_temp(self, agent):
        """Test pressurized hot water for temperatures above 100°C"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=150,
            storage_duration_hours=8,
            storage_capacity_kwh=3000,
            application="load_shifting"
        )

        assert result["recommended_technology"] == "pressurized_hot_water"
        assert result["technology_specifications"]["capex_per_kwh_usd"] >= 30

    def test_molten_salt_selection(self, agent):
        """Test molten salt selection for high temperature"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=300,
            storage_duration_hours=12,
            storage_capacity_kwh=5000,
            application="solar_thermal",
            space_constraints="limited",
            budget_constraint="high"
        )

        assert result["recommended_technology"] == "molten_salt"
        assert result["technology_specifications"]["energy_density_kwh_m3"] > 100
        assert "salt" in result["technology_specifications"]["storage_medium"].lower()

    def test_thermochemical_very_high_temp(self, agent):
        """Test thermochemical storage for very high temperature"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=450,
            storage_duration_hours=6,
            storage_capacity_kwh=4000,
            application="backup"
        )

        assert result["recommended_technology"] == "thermochemical"
        assert result["technology_specifications"]["energy_density_kwh_m3"] >= 150

    def test_steam_accumulator_short_duration(self, agent):
        """Test steam accumulator selection for short duration > 100°C"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=120,
            storage_duration_hours=2,
            storage_capacity_kwh=1000,
            application="demand_response"
        )

        # Short duration at >100°C should recommend steam accumulator
        assert result["recommended_technology"] in ["steam_accumulator", "pressurized_hot_water"]

    def test_alternative_technologies_provided(self, agent):
        """Test that alternative technologies are provided"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=90,
            storage_duration_hours=6,
            storage_capacity_kwh=2500,
            application="solar_thermal"
        )

        assert "alternative_technologies" in result
        # Hot water tank should have PCM as alternative
        if result["recommended_technology"] == "hot_water_tank":
            assert len(result["alternative_technologies"]) > 0

    def test_technology_rationale_format(self, agent):
        """Test that technology rationale is well-formatted"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=85,
            storage_duration_hours=8,
            storage_capacity_kwh=3000,
            application="load_shifting"
        )

        assert "technology_rationale" in result
        assert len(result["technology_rationale"]) > 20  # Should be descriptive
        assert "°C" in result["technology_rationale"] or "temperature" in result["technology_rationale"].lower()

    def test_design_considerations_completeness(self, agent):
        """Test that design considerations are comprehensive"""
        result = agent._select_storage_technology_impl(
            operating_temperature_c=250,
            storage_duration_hours=10,
            storage_capacity_kwh=4000,
            application="solar_thermal"
        )

        assert len(result["design_considerations"]) >= 3
        # Molten salt should have freeze protection consideration
        if result["recommended_technology"] == "molten_salt":
            assert any("freeze" in c.lower() for c in result["design_considerations"])


# ============================================================================
# CATEGORY 4: TOOL 3 - OPTIMIZE CHARGE DISCHARGE (7 tests)
# ============================================================================

class TestOptimizeChargeDischarge:
    """Test charge/discharge optimization tool"""

    def test_solar_thermal_optimization(self, agent):
        """Test optimization for solar thermal charging"""
        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=2667,
            charge_source="solar_thermal",
            charge_power_kw=600,
            discharge_power_kw=500
        )

        assert "daily_energy_throughput_kwh" in result
        assert "solar_fraction" in result
        assert result["solar_fraction"] > 0
        assert "solar" in result["optimization_strategy"].lower()
        assert result["average_round_trip_efficiency"] > 0

    def test_load_shifting_optimization(self, agent):
        """Test optimization for load shifting"""
        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=2000,
            charge_source="electric_heater",
            charge_power_kw=400,
            discharge_power_kw=350
        )

        assert result["cost_savings_usd_per_day"] > 0
        assert result["annual_cost_savings_usd"] == result["cost_savings_usd_per_day"] * 365
        assert "peak" in result["optimization_strategy"].lower()

    def test_custom_tou_pricing(self, agent):
        """Test optimization with custom TOU pricing"""
        tou_pricing = [
            {"hour": h, "price_usd_per_kwh": 0.05 if h < 8 else 0.18}
            for h in range(24)
        ]

        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=1500,
            charge_source="grid_electric",
            charge_power_kw=300,
            discharge_power_kw=250,
            time_of_use_pricing=tou_pricing
        )

        assert result["cost_savings_usd_per_day"] > 0
        # Should mention pricing in strategy
        assert "$" in result["optimization_strategy"]

    def test_solar_availability_custom(self, agent):
        """Test solar optimization with custom availability"""
        solar_hours = [0] * 6 + [0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 0.9, 0.6, 0.2] + [0] * 9

        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=3000,
            charge_source="solar_thermal",
            charge_power_kw=700,
            discharge_power_kw=500,
            solar_availability_hours=solar_hours
        )

        assert result["solar_fraction"] > 0.5
        assert result["daily_energy_throughput_kwh"] > 0

    def test_waste_heat_source(self, agent):
        """Test optimization for waste heat recovery"""
        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=1000,
            charge_source="waste_heat",
            charge_power_kw=200,
            discharge_power_kw=150
        )

        assert result["average_round_trip_efficiency"] > 0
        # Waste heat doesn't have direct cost savings like TOU
        assert "daily_energy_throughput_kwh" in result

    def test_heat_pump_charging(self, agent):
        """Test optimization for heat pump charging"""
        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=2500,
            charge_source="heat_pump",
            charge_power_kw=500,
            discharge_power_kw=450
        )

        assert result["daily_energy_throughput_kwh"] > 0
        assert result["average_round_trip_efficiency"] <= 1.0

    def test_annual_savings_calculation(self, agent):
        """Test annual savings extrapolation"""
        result = agent._optimize_charge_discharge_impl(
            storage_capacity_kwh=2000,
            charge_source="electric_heater",
            charge_power_kw=400,
            discharge_power_kw=350
        )

        expected_annual = result["cost_savings_usd_per_day"] * 365
        assert abs(result["annual_cost_savings_usd"] - expected_annual) < 1


# ============================================================================
# CATEGORY 5: TOOL 4 - CALCULATE THERMAL LOSSES (7 tests)
# ============================================================================

class TestCalculateThermalLosses:
    """Test thermal loss calculation tool"""

    def test_basic_heat_loss_calculation(self, agent):
        """Test basic heat loss calculation"""
        result = agent._calculate_thermal_losses_impl(
            storage_volume_m3=60,
            storage_temperature_c=90,
            ambient_temperature_c=20,
            insulation_type="polyurethane_3inch",
            geometry="cylindrical_vertical",
            insulation_condition="good"
        )

        assert "surface_area_m2" in result
        assert "u_value_w_m2k" in result
        assert "heat_loss_rate_kw" in result
        assert "daily_energy_loss_kwh" in result
        assert result["heat_loss_rate_kw"] > 0
        assert result["daily_energy_loss_kwh"] == result["heat_loss_rate_kw"] * 24

    def test_insulation_comparison(self, agent):
        """Test different insulation types"""
        no_insulation = agent._calculate_thermal_losses_impl(
            storage_volume_m3=50,
            storage_temperature_c=80,
            insulation_type="none",
            geometry="cylindrical_vertical"
        )

        good_insulation = agent._calculate_thermal_losses_impl(
            storage_volume_m3=50,
            storage_temperature_c=80,
            insulation_type="polyurethane_6inch",
            geometry="cylindrical_vertical"
        )

        # Good insulation should have much lower losses
        assert good_insulation["heat_loss_rate_kw"] < no_insulation["heat_loss_rate_kw"] * 0.2

    def test_geometry_variations(self, agent):
        """Test different storage geometries"""
        cylindrical = agent._calculate_thermal_losses_impl(
            storage_volume_m3=100,
            storage_temperature_c=90,
            insulation_type="polyurethane_3inch",
            geometry="cylindrical_vertical"
        )

        spherical = agent._calculate_thermal_losses_impl(
            storage_volume_m3=100,
            storage_temperature_c=90,
            insulation_type="polyurethane_3inch",
            geometry="spherical"
        )

        # Sphere has minimum surface area for given volume
        assert spherical["surface_area_m2"] <= cylindrical["surface_area_m2"]

    def test_insulation_condition_impact(self, agent):
        """Test insulation condition impact on losses"""
        excellent = agent._calculate_thermal_losses_impl(
            storage_volume_m3=75,
            storage_temperature_c=85,
            insulation_type="polyurethane_3inch",
            insulation_condition="excellent"
        )

        poor = agent._calculate_thermal_losses_impl(
            storage_volume_m3=75,
            storage_temperature_c=85,
            insulation_type="polyurethane_3inch",
            insulation_condition="poor"
        )

        # Poor condition should have higher U-value and losses
        assert poor["u_value_w_m2k"] > excellent["u_value_w_m2k"]
        assert poor["heat_loss_rate_kw"] > excellent["heat_loss_rate_kw"]

    def test_temperature_decay_calculation(self, agent):
        """Test temperature decay calculation"""
        result = agent._calculate_thermal_losses_impl(
            storage_volume_m3=80,
            storage_temperature_c=95,
            ambient_temperature_c=20,
            insulation_type="polyurethane_3inch"
        )

        assert "temperature_decay_per_hour_k" in result
        assert result["temperature_decay_per_hour_k"] > 0
        # Should be a small decay rate with good insulation
        assert result["temperature_decay_per_hour_k"] < 1.0

    def test_annual_standby_loss(self, agent):
        """Test annual standby loss calculation"""
        result = agent._calculate_thermal_losses_impl(
            storage_volume_m3=100,
            storage_temperature_c=90,
            insulation_type="fiberglass_4inch"
        )

        # Annual should be daily × 365 / 1000 (to convert to MWh)
        expected_annual_mwh = result["daily_energy_loss_kwh"] * 365 / 1000
        assert abs(result["annual_standby_loss_mwh"] - expected_annual_mwh) < 0.1

    def test_insulation_upgrade_recommendations(self, agent):
        """Test insulation upgrade recommendations"""
        result = agent._calculate_thermal_losses_impl(
            storage_volume_m3=90,
            storage_temperature_c=85,
            insulation_type="fiberglass_2inch"
        )

        # Poor insulation should have upgrade recommendations
        assert len(result["insulation_upgrade_recommendations"]) > 0
        upgrade = result["insulation_upgrade_recommendations"][0]
        assert "upgrade" in upgrade
        assert "additional_cost_usd" in upgrade
        assert "loss_reduction_percent" in upgrade
        assert "payback_years" in upgrade


# ============================================================================
# CATEGORY 6: TOOL 5 - INTEGRATE WITH SOLAR (7 tests)
# ============================================================================

class TestIntegrateWithSolar:
    """Test solar thermal integration tool"""

    def test_basic_solar_integration(self, agent):
        """Test basic solar + storage integration"""
        result = agent._integrate_with_solar_impl(
            process_thermal_load_kw=400,
            process_temperature_c=90,
            latitude=35.0,
            annual_irradiance_kwh_m2=1850,
            load_profile="continuous_24x7",
            collector_type="flat_plate",
            storage_hours_target=8
        )

        assert "collector_area_m2" in result
        assert "storage_capacity_kwh" in result
        assert "solar_fraction_with_storage" in result
        assert "solar_fraction_no_storage" in result
        assert result["solar_fraction_with_storage"] > result["solar_fraction_no_storage"]

    def test_solar_fraction_improvement(self, agent):
        """Test solar fraction improvement calculation"""
        result = agent._integrate_with_solar_impl(
            process_thermal_load_kw=500,
            process_temperature_c=95,
            latitude=30.0,
            annual_irradiance_kwh_m2=2000,
            load_profile="continuous_24x7",
            collector_type="evacuated_tube",
            storage_hours_target=6
        )

        # Storage should provide significant boost
        improvement = result["solar_fraction_improvement_percent"]
        assert improvement > 0
        # Check calculation
        expected_improvement = (
            (result["solar_fraction_with_storage"] - result["solar_fraction_no_storage"])
            / result["solar_fraction_no_storage"] * 100
        )
        assert abs(improvement - expected_improvement) < 1

    def test_collector_type_comparison(self, agent):
        """Test different collector types"""
        flat_plate = agent._integrate_with_solar_impl(
            process_thermal_load_kw=300,
            process_temperature_c=70,
            latitude=40.0,
            annual_irradiance_kwh_m2=1600,
            load_profile="daytime_only",
            collector_type="flat_plate",
            storage_hours_target=4
        )

        parabolic = agent._integrate_with_solar_impl(
            process_thermal_load_kw=300,
            process_temperature_c=150,
            latitude=40.0,
            annual_irradiance_kwh_m2=1600,
            load_profile="daytime_only",
            collector_type="parabolic_trough",
            storage_hours_target=4
        )

        # Higher temp needs parabolic, should cost more
        assert parabolic["system_capex_usd"] > flat_plate["system_capex_usd"]

    def test_load_profile_impact(self, agent):
        """Test impact of different load profiles"""
        continuous = agent._integrate_with_solar_impl(
            process_thermal_load_kw=400,
            process_temperature_c=85,
            latitude=35.0,
            annual_irradiance_kwh_m2=1800,
            load_profile="continuous_24x7",
            collector_type="flat_plate",
            storage_hours_target=8
        )

        daytime = agent._integrate_with_solar_impl(
            process_thermal_load_kw=400,
            process_temperature_c=85,
            latitude=35.0,
            annual_irradiance_kwh_m2=1800,
            load_profile="daytime_only",
            collector_type="flat_plate",
            storage_hours_target=8
        )

        # Daytime-only should have higher solar fraction without storage
        assert daytime["solar_fraction_no_storage"] > continuous["solar_fraction_no_storage"]

    def test_storage_duration_impact(self, agent):
        """Test impact of storage duration on solar fraction"""
        short_storage = agent._integrate_with_solar_impl(
            process_thermal_load_kw=350,
            process_temperature_c=90,
            latitude=32.0,
            annual_irradiance_kwh_m2=1900,
            load_profile="continuous_24x7",
            collector_type="flat_plate",
            storage_hours_target=4
        )

        long_storage = agent._integrate_with_solar_impl(
            process_thermal_load_kw=350,
            process_temperature_c=90,
            latitude=32.0,
            annual_irradiance_kwh_m2=1900,
            load_profile="continuous_24x7",
            collector_type="flat_plate",
            storage_hours_target=12
        )

        # Longer storage should achieve higher solar fraction
        assert long_storage["solar_fraction_with_storage"] > short_storage["solar_fraction_with_storage"]
        assert long_storage["storage_capacity_kwh"] > short_storage["storage_capacity_kwh"]

    def test_capex_breakdown(self, agent):
        """Test that CAPEX is reasonable"""
        result = agent._integrate_with_solar_impl(
            process_thermal_load_kw=500,
            process_temperature_c=95,
            latitude=35.0,
            annual_irradiance_kwh_m2=1850,
            load_profile="continuous_24x7",
            collector_type="evacuated_tube",
            storage_hours_target=8
        )

        # CAPEX should be positive and reasonable
        assert result["system_capex_usd"] > 0
        assert result["system_capex_usd"] < 10000000  # Not absurdly high

    def test_design_recommendations_provided(self, agent):
        """Test that design recommendations are provided"""
        result = agent._integrate_with_solar_impl(
            process_thermal_load_kw=450,
            process_temperature_c=88,
            latitude=25.0,
            annual_irradiance_kwh_m2=2100,
            load_profile="continuous_24x7",
            collector_type="flat_plate",
            storage_hours_target=6
        )

        assert len(result["design_recommendations"]) >= 3
        # Should mention solar fraction
        assert any("solar fraction" in r.lower() for r in result["design_recommendations"])


# ============================================================================
# CATEGORY 7: TOOL 6 - CALCULATE ECONOMICS (8 tests)
# ============================================================================

class TestCalculateEconomics:
    """Test economic analysis tool"""

    def test_basic_economics_calculation(self, agent):
        """Test basic economic analysis"""
        result = agent._calculate_economics_impl(
            storage_capacity_kwh=2667,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1750000,
            energy_cost_usd_per_kwh=0.08
        )

        assert "capex_usd" in result
        assert "annual_savings_usd" in result
        assert "simple_payback_years" in result
        assert "npv_usd" in result
        assert "irr" in result
        assert "financial_rating" in result

    def test_excellent_payback_rating(self, agent):
        """Test excellent payback rating (<3 years)"""
        result = agent._calculate_economics_impl(
            storage_capacity_kwh=2000,
            technology="hot_water_tank",
            annual_energy_savings_kwh=2000000,
            energy_cost_usd_per_kwh=0.10,
            demand_charge_savings_usd_per_month=1000
        )

        assert result["simple_payback_years"] < 3
        assert result["financial_rating"] == "Excellent (<3yr payback)"

    def test_demand_charge_savings_impact(self, agent):
        """Test impact of demand charge savings"""
        without_demand = agent._calculate_economics_impl(
            storage_capacity_kwh=3000,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1500000,
            energy_cost_usd_per_kwh=0.08,
            demand_charge_savings_usd_per_month=0
        )

        with_demand = agent._calculate_economics_impl(
            storage_capacity_kwh=3000,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1500000,
            energy_cost_usd_per_kwh=0.08,
            demand_charge_savings_usd_per_month=800
        )

        # Demand charges should increase annual savings
        assert with_demand["annual_savings_usd"] > without_demand["annual_savings_usd"]
        assert with_demand["simple_payback_years"] < without_demand["simple_payback_years"]

    def test_incentives_impact(self, agent):
        """Test impact of incentives on payback"""
        without_incentive = agent._calculate_economics_impl(
            storage_capacity_kwh=2500,
            technology="pressurized_hot_water",
            annual_energy_savings_kwh=1200000,
            energy_cost_usd_per_kwh=0.09,
            incentives_usd=0
        )

        with_incentive = agent._calculate_economics_impl(
            storage_capacity_kwh=2500,
            technology="pressurized_hot_water",
            annual_energy_savings_kwh=1200000,
            energy_cost_usd_per_kwh=0.09,
            incentives_usd=25000
        )

        # Incentives should improve payback
        assert with_incentive["simple_payback_years"] < without_incentive["simple_payback_years"]

    def test_technology_cost_defaults(self, agent):
        """Test default costs for different technologies"""
        hot_water = agent._calculate_economics_impl(
            storage_capacity_kwh=2000,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1000000,
            energy_cost_usd_per_kwh=0.08
        )

        pcm = agent._calculate_economics_impl(
            storage_capacity_kwh=2000,
            technology="pcm",
            annual_energy_savings_kwh=1000000,
            energy_cost_usd_per_kwh=0.08
        )

        # PCM should be more expensive
        assert pcm["capex_per_kwh_usd"] > hot_water["capex_per_kwh_usd"]
        assert pcm["capex_usd"] > hot_water["capex_usd"]

    def test_npv_calculation(self, agent):
        """Test NPV calculation accuracy"""
        result = agent._calculate_economics_impl(
            storage_capacity_kwh=2500,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1500000,
            energy_cost_usd_per_kwh=0.08,
            system_lifetime_years=25,
            discount_rate=0.06
        )

        # NPV should be positive for good projects
        if result["simple_payback_years"] < 10:
            assert result["npv_usd"] > 0

    def test_opex_calculation(self, agent):
        """Test OPEX calculation"""
        result = agent._calculate_economics_impl(
            storage_capacity_kwh=3000,
            technology="molten_salt",
            annual_energy_savings_kwh=1800000,
            energy_cost_usd_per_kwh=0.10,
            opex_percent_of_capex=0.02  # 2% O&M
        )

        expected_opex = result["capex_usd"] * 0.02
        assert abs(result["annual_opex_usd"] - expected_opex) < 1

    def test_lcoe_storage_calculation(self, agent):
        """Test LCOE_storage calculation"""
        result = agent._calculate_economics_impl(
            storage_capacity_kwh=2000,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1500000,
            energy_cost_usd_per_kwh=0.08,
            system_lifetime_years=25,
            discount_rate=0.06
        )

        # LCOE should be positive and reasonable
        assert result["lcoe_storage_usd_per_kwh"] > 0
        assert result["lcoe_storage_usd_per_kwh"] < 0.10  # Should be competitive


# ============================================================================
# CATEGORY 8: INTEGRATION TESTS (4 tests)
# ============================================================================

class TestIntegration:
    """Test full agent integration"""

    def test_validate_solar_thermal_payload(self, agent, solar_thermal_payload):
        """Test validation of solar thermal payload"""
        assert agent.validate(solar_thermal_payload) is True

    def test_validate_load_shifting_payload(self, agent, load_shifting_payload):
        """Test validation of load shifting payload"""
        assert agent.validate(load_shifting_payload) is True

    def test_missing_required_field(self, agent):
        """Test validation fails for missing required field"""
        incomplete_payload = {
            "application": "solar_thermal",
            "thermal_load_kw": 400,
            # Missing other required fields
        }
        with pytest.raises(ValueError, match="Missing required field"):
            agent.validate(incomplete_payload)

    def test_invalid_temperature_range(self, agent):
        """Test validation fails for invalid temperature"""
        invalid_payload = {
            "application": "solar_thermal",
            "thermal_load_kw": 400,
            "temperature_c": 500,  # Too high
            "storage_hours": 8,
            "load_profile": "continuous_24x7",
            "energy_cost_usd_per_kwh": 0.08,
        }
        with pytest.raises(ValueError, match="temperature_c must be between"):
            agent.validate(invalid_payload)


# ============================================================================
# CATEGORY 9: DETERMINISM TESTS (3 tests)
# ============================================================================

class TestDeterminism:
    """Test deterministic behavior"""

    def test_storage_capacity_determinism(self, agent):
        """Test that storage capacity calculation is deterministic"""
        result1 = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=500,
            average_thermal_load_kw=400,
            storage_duration_hours=6,
            operating_temperature_c=90,
            return_temperature_c=50
        )

        result2 = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=500,
            average_thermal_load_kw=400,
            storage_duration_hours=6,
            operating_temperature_c=90,
            return_temperature_c=50
        )

        # Should produce identical results
        assert result1 == result2
        assert result1["storage_capacity_kwh"] == result2["storage_capacity_kwh"]
        assert result1["volume_storage_medium_m3"] == result2["volume_storage_medium_m3"]

    def test_technology_selection_determinism(self, agent):
        """Test that technology selection is deterministic"""
        result1 = agent._select_storage_technology_impl(
            operating_temperature_c=90,
            storage_duration_hours=6,
            storage_capacity_kwh=2667,
            application="solar_thermal"
        )

        result2 = agent._select_storage_technology_impl(
            operating_temperature_c=90,
            storage_duration_hours=6,
            storage_capacity_kwh=2667,
            application="solar_thermal"
        )

        assert result1["recommended_technology"] == result2["recommended_technology"]
        assert result1["technology_rationale"] == result2["technology_rationale"]

    def test_economics_determinism(self, agent):
        """Test that economics calculation is deterministic"""
        result1 = agent._calculate_economics_impl(
            storage_capacity_kwh=2667,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1750000,
            energy_cost_usd_per_kwh=0.08
        )

        result2 = agent._calculate_economics_impl(
            storage_capacity_kwh=2667,
            technology="hot_water_tank",
            annual_energy_savings_kwh=1750000,
            energy_cost_usd_per_kwh=0.08
        )

        assert result1["simple_payback_years"] == result2["simple_payback_years"]
        assert result1["npv_usd"] == result2["npv_usd"]


# ============================================================================
# CATEGORY 10: ERROR HANDLING TESTS (6 tests)
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_negative_thermal_load(self, agent):
        """Test error handling for negative thermal load"""
        invalid_payload = {
            "application": "solar_thermal",
            "thermal_load_kw": -100,  # Negative
            "temperature_c": 90,
            "storage_hours": 8,
            "load_profile": "continuous_24x7",
            "energy_cost_usd_per_kwh": 0.08,
        }
        with pytest.raises(ValueError, match="must be positive"):
            agent.validate(invalid_payload)

    def test_invalid_storage_hours(self, agent):
        """Test error handling for invalid storage hours"""
        invalid_payload = {
            "application": "solar_thermal",
            "thermal_load_kw": 400,
            "temperature_c": 90,
            "storage_hours": 200,  # Too high
            "load_profile": "continuous_24x7",
            "energy_cost_usd_per_kwh": 0.08,
        }
        with pytest.raises(ValueError, match="storage_hours must be between"):
            agent.validate(invalid_payload)

    def test_zero_energy_savings(self, agent):
        """Test handling of zero energy savings"""
        result = agent._calculate_economics_impl(
            storage_capacity_kwh=2000,
            technology="hot_water_tank",
            annual_energy_savings_kwh=0,  # Zero savings
            energy_cost_usd_per_kwh=0.08
        )

        # Should handle gracefully
        assert result["annual_savings_usd"] == 0
        assert result["simple_payback_years"] == 999  # Infinite payback

    def test_unknown_tool_call(self, agent):
        """Test error handling for unknown tool"""
        with pytest.raises(ValueError, match="Unknown tool"):
            agent._handle_tool_call("nonexistent_tool", {})

    def test_very_small_storage_volume(self, agent):
        """Test handling of very small storage volumes"""
        result = agent._calculate_thermal_losses_impl(
            storage_volume_m3=1,  # Very small
            storage_temperature_c=80,
            insulation_type="polyurethane_3inch"
        )

        # Should still calculate reasonable values
        assert result["surface_area_m2"] > 0
        assert result["heat_loss_rate_kw"] >= 0

    def test_very_low_temperature_delta(self, agent):
        """Test handling of very low temperature delta"""
        result = agent._calculate_storage_capacity_impl(
            peak_thermal_load_kw=400,
            average_thermal_load_kw=350,
            storage_duration_hours=6,
            operating_temperature_c=55,
            return_temperature_c=50,  # Only 5K delta
            round_trip_efficiency=0.90
        )

        # Should handle low delta gracefully
        assert result["temperature_delta_k"] == 5
        # Volume should be very large due to low delta
        assert result["volume_storage_medium_m3"] > 100


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=greenlang.agents.thermal_storage_agent_ai", "--cov-report=term-missing"])
