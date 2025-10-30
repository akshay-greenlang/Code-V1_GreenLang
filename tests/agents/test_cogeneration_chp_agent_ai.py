"""
Unit tests for CogenerationCHPAgent_AI

Test Coverage:
--------------
1. Configuration Tests (5 tests)
2. Tool 1: select_chp_technology (8 tests)
3. Tool 2: calculate_chp_performance (8 tests)
4. Tool 3: size_heat_recovery_system (6 tests)
5. Tool 4: calculate_economic_metrics (8 tests)
6. Tool 5: assess_grid_interconnection (7 tests)
7. Tool 6: optimize_operating_strategy (6 tests)
8. Tool 7: calculate_emissions_reduction (6 tests)
9. Tool 8: generate_chp_report (4 tests)
10. Integration Tests (3 tests)
11. Determinism Tests (3 tests)
12. Error Handling Tests (6 tests)

Total: 70+ tests
Target Coverage: 85%+

Standards:
---------
- EPA CHP Partnership
- ASHRAE CHP Applications
- IEEE 1547-2018
- ASME BPVC Section I
- NIST 135 Economic Analysis
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime

from greenlang.agents.cogeneration_chp_agent_ai import (
    CogenerationCHPAgentAI,
    CogenerationCHPConfig,
    CHPTechnologyDatabase
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def default_config() -> CogenerationCHPConfig:
    """Default agent configuration for testing"""
    return CogenerationCHPConfig(
        agent_id="test/cogeneration_chp_agent",
        agent_name="TestCogenerationCHPAgent",
        budget_usd=0.50,
        temperature=0.0,
        seed=42,
        deterministic=True
    )


@pytest.fixture
def agent(default_config) -> CogenerationCHPAgentAI:
    """Agent instance for testing"""
    return CogenerationCHPAgentAI(config=default_config)


@pytest.fixture
def tech_database() -> CHPTechnologyDatabase:
    """Technology database for testing"""
    return CHPTechnologyDatabase()


# ============================================================================
# CATEGORY 1: CONFIGURATION TESTS (5 tests)
# ============================================================================

class TestConfiguration:
    """Test agent configuration and initialization"""

    def test_default_configuration(self, default_config):
        """Test default configuration values"""
        assert default_config.agent_id == "test/cogeneration_chp_agent"
        assert default_config.agent_name == "TestCogenerationCHPAgent"
        assert default_config.budget_usd == 0.50
        assert default_config.temperature == 0.0
        assert default_config.seed == 42
        assert default_config.deterministic is True

    def test_configuration_immutability(self, default_config):
        """Test that configuration is immutable"""
        with pytest.raises(Exception):
            default_config.temperature = 0.5

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.config.agent_name == "TestCogenerationCHPAgent"
        assert agent.tech_db is not None
        assert agent.tool_registry is not None
        assert agent.provenance is not None

    def test_agent_version(self, agent):
        """Test agent version is set"""
        version = agent._version()
        assert version == "1.0.0"

    def test_tool_registration(self, agent):
        """Test all 8 tools are registered"""
        tools = agent.tool_registry.get_all()
        assert len(tools) == 8
        assert "select_chp_technology" in [t.name for t in tools]
        assert "calculate_chp_performance" in [t.name for t in tools]
        assert "size_heat_recovery_system" in [t.name for t in tools]
        assert "calculate_economic_metrics" in [t.name for t in tools]
        assert "assess_grid_interconnection" in [t.name for t in tools]
        assert "optimize_operating_strategy" in [t.name for t in tools]
        assert "calculate_emissions_reduction" in [t.name for t in tools]
        assert "generate_chp_report" in [t.name for t in tools]


# ============================================================================
# CATEGORY 2: TOOL 1 - SELECT CHP TECHNOLOGY (8 tests)
# ============================================================================

class TestSelectCHPTechnology:
    """Test CHP technology selection tool"""

    def test_reciprocating_engine_selection(self, agent):
        """Test selection of reciprocating engine for balanced loads"""
        result = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )

        assert result["recommended_technology"] == "reciprocating_engine"
        assert result["typical_electrical_efficiency"] >= 0.35
        assert result["typical_total_efficiency"] >= 0.75
        assert result["deterministic"] is True
        assert "provenance" in result

    def test_gas_turbine_selection_large_facility(self, agent):
        """Test gas turbine selection for large facilities"""
        result = agent.select_chp_technology(
            electrical_demand_kw=5000,
            thermal_demand_mmbtu_hr=25.0,
            heat_to_power_ratio=1.5,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="standard",
            space_constraints="ample"
        )

        # Should recommend gas turbine or reciprocating engine for large size
        assert result["recommended_technology"] in ["gas_turbine", "reciprocating_engine"]
        assert result["deterministic"] is True

    def test_microturbine_selection_small_facility(self, agent):
        """Test microturbine selection for small facilities"""
        result = agent.select_chp_technology(
            electrical_demand_kw=200,
            thermal_demand_mmbtu_hr=1.5,
            heat_to_power_ratio=2.2,
            load_profile_type="daytime_only",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="very_limited"
        )

        assert result["recommended_technology"] == "microturbine"
        assert result["typical_electrical_efficiency"] >= 0.26
        assert result["deterministic"] is True

    def test_fuel_cell_high_efficiency_requirement(self, agent):
        """Test fuel cell selection when high electrical efficiency required"""
        result = agent.select_chp_technology(
            electrical_demand_kw=1000,
            thermal_demand_mmbtu_hr=5.0,
            heat_to_power_ratio=1.5,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="ultra_low",
            space_constraints="moderate",
            required_electrical_efficiency=0.40
        )

        # Fuel cell has highest electrical efficiency
        assert result["recommended_technology"] == "fuel_cell"
        assert result["typical_electrical_efficiency"] >= 0.40
        assert result["deterministic"] is True

    def test_steam_turbine_high_thermal_ratio(self, agent):
        """Test steam turbine selection for high thermal loads"""
        result = agent.select_chp_technology(
            electrical_demand_kw=3000,
            thermal_demand_mmbtu_hr=70.0,
            heat_to_power_ratio=6.8,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas", "biogas"],
            emissions_requirements="standard",
            space_constraints="ample"
        )

        # Steam turbine best for very high H/P ratios
        assert result["recommended_technology"] == "steam_turbine"
        assert result["heat_to_power_ratio_achievable"] >= 4.0
        assert result["deterministic"] is True

    def test_technology_selection_scoring(self, agent):
        """Test that selection returns scoring reasons"""
        result = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )

        assert "selection_score" in result
        assert result["selection_score"] > 0
        assert "selection_reasons" in result
        assert len(result["selection_reasons"]) > 0

    def test_alternative_technology_provided(self, agent):
        """Test that alternative technology is provided"""
        result = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )

        assert "alternative_technology" in result
        assert result["alternative_technology"] is not None
        assert result["alternative_technology"] != result["recommended_technology"]

    def test_invalid_input_handling(self, agent):
        """Test error handling for invalid inputs"""
        with pytest.raises(ValueError, match="electrical_demand_kw must be positive"):
            agent.select_chp_technology(
                electrical_demand_kw=-100,
                thermal_demand_mmbtu_hr=15.0,
                heat_to_power_ratio=2.2,
                load_profile_type="baseload_24x7",
                available_fuels=["natural_gas"],
                emissions_requirements="low_nox",
                space_constraints="moderate"
            )


# ============================================================================
# CATEGORY 3: TOOL 2 - CALCULATE CHP PERFORMANCE (8 tests)
# ============================================================================

class TestCalculateCHPPerformance:
    """Test CHP performance calculation tool"""

    def test_reciprocating_engine_performance(self, agent):
        """Test performance calculation for reciprocating engine"""
        result = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )

        assert result["electrical_output_kw"] == 2000
        assert 0.35 <= result["electrical_efficiency"] <= 0.45
        assert 0.75 <= result["total_efficiency"] <= 0.90
        assert result["deterministic"] is True
        assert "provenance" in result

    def test_gas_turbine_performance(self, agent):
        """Test performance calculation for gas turbine"""
        result = agent.calculate_chp_performance(
            chp_technology="gas_turbine",
            electrical_capacity_kw=5000,
            fuel_input_mmbtu_hr=50.0,
            heat_recovery_configuration="hrsg_unfired",
            exhaust_temperature_f=1050,
            exhaust_mass_flow_lb_hr=80000,
            heat_recovery_target_temperature_f=400
        )

        assert result["electrical_output_kw"] == 5000
        assert 0.25 <= result["electrical_efficiency"] <= 0.42
        assert result["exhaust_energy_available_mmbtu_hr"] > 0
        assert result["deterministic"] is True

    def test_part_load_performance_recip_engine(self, agent):
        """Test part-load performance derating for reciprocating engine"""
        result_full = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250,
            part_load_ratio=1.0
        )

        result_part = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250,
            part_load_ratio=0.75
        )

        # Part load should have lower output but minimal efficiency penalty for recip engines
        assert result_part["electrical_output_kw"] < result_full["electrical_output_kw"]
        assert result_part["part_load_penalty_pct"] < 5.0  # Recip engines have <3% penalty

    def test_part_load_performance_gas_turbine(self, agent):
        """Test part-load performance derating for gas turbine"""
        result_part = agent.calculate_chp_performance(
            chp_technology="gas_turbine",
            electrical_capacity_kw=5000,
            fuel_input_mmbtu_hr=50.0,
            heat_recovery_configuration="hrsg_unfired",
            exhaust_temperature_f=1050,
            exhaust_mass_flow_lb_hr=80000,
            heat_recovery_target_temperature_f=400,
            part_load_ratio=0.70
        )

        # Gas turbines have significant part-load penalty
        assert result_part["part_load_penalty_pct"] > 5.0
        assert result_part["electrical_output_kw"] == 5000 * 0.70

    def test_heat_rate_calculation(self, agent):
        """Test heat rate calculation"""
        result = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )

        # Heat rate should be in reasonable range (8000-10000 Btu/kWh for good CHP)
        assert 7000 <= result["heat_rate_btu_per_kwh"] <= 12000

    def test_thermal_output_calculation(self, agent):
        """Test thermal output calculation"""
        result = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )

        assert result["thermal_output_mmbtu_hr"] > 0
        assert result["thermal_efficiency"] > 0
        assert result["heat_recovery_effectiveness"] > 0

    def test_fuel_cell_high_efficiency(self, agent):
        """Test fuel cell achieves high electrical efficiency"""
        result = agent.calculate_chp_performance(
            chp_technology="fuel_cell",
            electrical_capacity_kw=1000,
            fuel_input_mmbtu_hr=7.5,
            heat_recovery_configuration="jacket_water_only",
            exhaust_temperature_f=550,
            exhaust_mass_flow_lb_hr=8000,
            heat_recovery_target_temperature_f=200
        )

        # Fuel cells should achieve 40%+ electrical efficiency
        assert result["electrical_efficiency"] >= 0.35

    def test_invalid_technology_error(self, agent):
        """Test error handling for invalid technology"""
        with pytest.raises(ValueError, match="chp_technology must be one of"):
            agent.calculate_chp_performance(
                chp_technology="invalid_tech",
                electrical_capacity_kw=2000,
                fuel_input_mmbtu_hr=18.0,
                heat_recovery_configuration="jacket_exhaust",
                exhaust_temperature_f=850,
                exhaust_mass_flow_lb_hr=25000,
                heat_recovery_target_temperature_f=250
            )


# ============================================================================
# CATEGORY 4: TOOL 3 - SIZE HEAT RECOVERY SYSTEM (6 tests)
# ============================================================================

class TestSizeHeatRecoverySystem:
    """Test heat recovery system sizing tool"""

    def test_hrsg_sizing_basic(self, agent):
        """Test basic HRSG sizing calculation"""
        result = agent.size_heat_recovery_system(
            exhaust_temperature_f=900,
            exhaust_mass_flow_lb_hr=30000,
            process_heat_demand_mmbtu_hr=12.0,
            process_temperature_requirement_f=350,
            heat_recovery_type="hrsg_unfired"
        )

        assert result["recovered_heat_mmbtu_hr"] > 0
        assert result["available_heat_mmbtu_hr"] > 0
        assert result["recovery_effectiveness_pct"] > 0
        assert result["deterministic"] is True

    def test_heat_recovery_limited_by_demand(self, agent):
        """Test that recovered heat is limited by process demand"""
        result = agent.size_heat_recovery_system(
            exhaust_temperature_f=1000,
            exhaust_mass_flow_lb_hr=50000,
            process_heat_demand_mmbtu_hr=10.0,  # Lower than available
            process_temperature_requirement_f=300,
            heat_recovery_type="hrsg_unfired"
        )

        # Recovered heat should not exceed demand
        assert result["recovered_heat_mmbtu_hr"] <= result["process_heat_demand_mmbtu_hr"]
        assert result["recovered_heat_mmbtu_hr"] <= result["available_heat_mmbtu_hr"]

    def test_high_temperature_process_heat(self, agent):
        """Test heat recovery for high temperature requirements"""
        result = agent.size_heat_recovery_system(
            exhaust_temperature_f=1100,
            exhaust_mass_flow_lb_hr=40000,
            process_heat_demand_mmbtu_hr=15.0,
            process_temperature_requirement_f=500,
            heat_recovery_type="hrsg_supplementary_fired"
        )

        # Stack temperature should be above process temperature + approach delta
        assert result["stack_temperature_f"] >= 500

    def test_heat_exchanger_area_calculation(self, agent):
        """Test heat exchanger area calculation"""
        result = agent.size_heat_recovery_system(
            exhaust_temperature_f=900,
            exhaust_mass_flow_lb_hr=30000,
            process_heat_demand_mmbtu_hr=12.0,
            process_temperature_requirement_f=350,
            heat_recovery_type="hrsg_unfired"
        )

        assert result["heat_exchanger_area_sqft"] > 0

    def test_cost_estimation(self, agent):
        """Test CAPEX estimation for heat recovery"""
        result = agent.size_heat_recovery_system(
            exhaust_temperature_f=900,
            exhaust_mass_flow_lb_hr=30000,
            process_heat_demand_mmbtu_hr=12.0,
            process_temperature_requirement_f=350,
            heat_recovery_type="hrsg_unfired"
        )

        assert result["estimated_capex_usd"] > 0
        # Should be roughly $75k per MMBtu/hr capacity
        assert 500000 <= result["estimated_capex_usd"] <= 1500000

    def test_invalid_inputs(self, agent):
        """Test error handling for invalid inputs"""
        with pytest.raises(ValueError, match="exhaust_temperature_f must be positive"):
            agent.size_heat_recovery_system(
                exhaust_temperature_f=-100,
                exhaust_mass_flow_lb_hr=30000,
                process_heat_demand_mmbtu_hr=12.0,
                process_temperature_requirement_f=350,
                heat_recovery_type="hrsg_unfired"
            )


# ============================================================================
# CATEGORY 5: TOOL 4 - CALCULATE ECONOMIC METRICS (8 tests)
# ============================================================================

class TestCalculateEconomicMetrics:
    """Test economic metrics calculation tool"""

    def test_spark_spread_calculation(self, agent):
        """Test spark spread calculation"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        assert "spark_spread_per_mwh" in result
        assert result["spark_spread_per_mwh"] > 0  # Should be positive with these rates
        assert result["deterministic"] is True

    def test_avoided_costs_calculation(self, agent):
        """Test avoided costs calculation"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        assert result["avoided_electricity_cost_annual"] > 0
        assert result["avoided_demand_charge_annual"] > 0
        assert result["avoided_thermal_cost_annual"] > 0
        assert result["total_avoided_costs_annual"] > 0

    def test_simple_payback_calculation(self, agent):
        """Test simple payback calculation"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        # Payback should be in reasonable range (3-8 years for typical CHP)
        assert 2 <= result["simple_payback_years"] <= 10

    def test_npv_calculation(self, agent):
        """Test NPV calculation over 20 years"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015,
            discount_rate=0.08
        )

        assert "npv_20yr" in result
        # NPV should be positive for good economics
        assert result["npv_20yr"] > 0

    def test_federal_itc_impact(self, agent):
        """Test impact of federal Investment Tax Credit"""
        result_no_itc = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015,
            federal_itc_percent=0.0
        )

        result_with_itc = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015,
            federal_itc_percent=10.0
        )

        # ITC should improve payback and NPV
        assert result_with_itc["simple_payback_years"] < result_no_itc["simple_payback_years"]
        assert result_with_itc["npv_20yr"] > result_no_itc["npv_20yr"]
        assert result_with_itc["federal_itc_value"] == 3_500_000 * 0.10

    def test_lcoe_calculation(self, agent):
        """Test levelized cost of electricity calculation"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        assert "lcoe_per_kwh" in result
        assert 0.03 <= result["lcoe_per_kwh"] <= 0.15  # Reasonable LCOE range

    def test_benefit_cost_ratio(self, agent):
        """Test benefit-cost ratio calculation"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        assert "benefit_cost_ratio" in result
        # BCR > 1.0 means project is worthwhile
        assert result["benefit_cost_ratio"] > 1.0

    def test_irr_calculation(self, agent):
        """Test internal rate of return calculation"""
        result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        assert "irr_percent" in result
        # IRR should be in reasonable range (10-25% for good projects)
        assert 5 <= result["irr_percent"] <= 30


# ============================================================================
# CATEGORY 6: TOOL 5 - ASSESS GRID INTERCONNECTION (7 tests)
# ============================================================================

class TestAssessGridInterconnection:
    """Test grid interconnection assessment tool"""

    def test_small_system_simplified_interconnection(self, agent):
        """Test small system gets simplified interconnection process"""
        result = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=20,
            facility_peak_demand_kw=100,
            voltage_level="low_voltage_480v",
            export_mode="no_export",
            utility_territory="investor_owned"
        )

        assert "Level 1: Simplified" in result["ieee_1547_category"]
        assert result["utility_application_timeline_weeks"] <= 4
        assert result["utility_study_required"] is False

    def test_medium_system_fast_track(self, agent):
        """Test medium system gets fast track process"""
        result = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=1500,
            facility_peak_demand_kw=3000,
            voltage_level="medium_voltage_4160v",
            export_mode="limited_export",
            utility_territory="investor_owned"
        )

        assert "Level 2: Fast Track" in result["ieee_1547_category"]
        assert result["utility_application_timeline_weeks"] <= 10
        assert result["deterministic"] is True

    def test_large_system_study_required(self, agent):
        """Test large system requires interconnection study"""
        result = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=5000,
            facility_peak_demand_kw=8000,
            voltage_level="medium_voltage_13kv",
            export_mode="full_export",
            utility_territory="investor_owned"
        )

        assert "Level 3: Study Process" in result["ieee_1547_category"]
        assert result["utility_study_required"] is True

    def test_interconnection_equipment_list(self, agent):
        """Test required interconnection equipment is listed"""
        result = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=2000,
            facility_peak_demand_kw=3500,
            voltage_level="medium_voltage_4160v",
            export_mode="limited_export",
            utility_territory="investor_owned"
        )

        assert len(result["required_equipment"]) > 0
        assert any("relay" in eq.lower() for eq in result["required_equipment"])
        assert any("disconnect" in eq.lower() for eq in result["required_equipment"])
        assert result["islanding_protection_required"] is True

    def test_standby_charges_by_utility_type(self, agent):
        """Test standby charges vary by utility type"""
        result_iou = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=2000,
            facility_peak_demand_kw=3500,
            voltage_level="low_voltage_480v",
            export_mode="no_export",
            utility_territory="investor_owned"
        )

        result_muni = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=2000,
            facility_peak_demand_kw=3500,
            voltage_level="low_voltage_480v",
            export_mode="no_export",
            utility_territory="municipal"
        )

        # IOUs typically have higher standby charges than munis
        assert result_iou["standby_charge_per_kw_month"] > result_muni["standby_charge_per_kw_month"]

    def test_export_compensation_rates(self, agent):
        """Test export compensation varies by export mode"""
        result_no_export = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=2000,
            facility_peak_demand_kw=3500,
            voltage_level="low_voltage_480v",
            export_mode="no_export",
            utility_territory="investor_owned"
        )

        result_full_export = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=2000,
            facility_peak_demand_kw=3500,
            voltage_level="low_voltage_480v",
            export_mode="full_export",
            utility_territory="investor_owned"
        )

        assert result_no_export["export_compensation_per_kwh"] == 0.0
        assert result_full_export["export_compensation_per_kwh"] > 0.0

    def test_grid_upgrade_cost_estimation(self, agent):
        """Test grid upgrade costs are estimated"""
        result = agent.assess_grid_interconnection(
            chp_electrical_capacity_kw=8000,
            facility_peak_demand_kw=10000,
            voltage_level="medium_voltage_13kv",
            export_mode="full_export",
            utility_territory="investor_owned",
            distance_to_substation_miles=2.0
        )

        # Large system with full export should have upgrade costs
        assert result["grid_upgrade_cost_estimate"] > 0
        assert result["total_interconnection_cost_estimate"] > result["estimated_interconnection_equipment_cost"]


# ============================================================================
# CATEGORY 7: TOOL 6 - OPTIMIZE OPERATING STRATEGY (6 tests)
# ============================================================================

class TestOptimizeOperatingStrategy:
    """Test operating strategy optimization tool"""

    def test_thermal_following_strategy(self, agent):
        """Test thermal-following operating strategy"""
        result = agent.optimize_operating_strategy(
            electrical_load_profile_kw=[2000] * 24,
            thermal_load_profile_mmbtu_hr=[15.0] * 24,
            chp_electrical_capacity_kw=2500,
            chp_thermal_capacity_mmbtu_hr=20.0,
            electricity_rate_schedule=[0.12] * 24,
            gas_price_per_mmbtu=6.0,
            strategy_type="thermal_following"
        )

        assert result["recommended_strategy"] in ["thermal_following", "baseload"]
        assert result["deterministic"] is True
        assert "annual_operating_hours" in result

    def test_electric_following_strategy(self, agent):
        """Test electric-following operating strategy"""
        result = agent.optimize_operating_strategy(
            electrical_load_profile_kw=[1800] * 24,
            thermal_load_profile_mmbtu_hr=[25.0] * 24,
            chp_electrical_capacity_kw=2000,
            chp_thermal_capacity_mmbtu_hr=15.0,
            electricity_rate_schedule=[0.12] * 24,
            gas_price_per_mmbtu=6.0,
            strategy_type="electric_following"
        )

        assert result["recommended_strategy"] in ["electric_following", "baseload"]
        assert result["deterministic"] is True

    def test_baseload_strategy(self, agent):
        """Test baseload operating strategy"""
        result = agent.optimize_operating_strategy(
            electrical_load_profile_kw=[2000] * 24,
            thermal_load_profile_mmbtu_hr=[15.0] * 24,
            chp_electrical_capacity_kw=2000,
            chp_thermal_capacity_mmbtu_hr=15.0,
            electricity_rate_schedule=[0.12] * 24,
            gas_price_per_mmbtu=6.0,
            strategy_type="baseload"
        )

        assert result["recommended_strategy"] == "baseload"
        assert 0 <= result["electrical_capacity_factor"] <= 1.0
        assert 0 <= result["thermal_capacity_factor"] <= 1.0

    def test_capacity_factor_calculation(self, agent):
        """Test capacity factor calculation"""
        result = agent.optimize_operating_strategy(
            electrical_load_profile_kw=[2000] * 24,
            thermal_load_profile_mmbtu_hr=[15.0] * 24,
            chp_electrical_capacity_kw=2000,
            chp_thermal_capacity_mmbtu_hr=15.0,
            electricity_rate_schedule=[0.12] * 24,
            gas_price_per_mmbtu=6.0,
            strategy_type="baseload"
        )

        # With constant load matching capacity, CF should be close to 1.0
        assert result["electrical_capacity_factor"] > 0.9
        assert result["thermal_capacity_factor"] > 0.9

    def test_annual_projections(self, agent):
        """Test annual generation projections"""
        result = agent.optimize_operating_strategy(
            electrical_load_profile_kw=[2000] * 24,
            thermal_load_profile_mmbtu_hr=[15.0] * 24,
            chp_electrical_capacity_kw=2000,
            chp_thermal_capacity_mmbtu_hr=15.0,
            electricity_rate_schedule=[0.12] * 24,
            gas_price_per_mmbtu=6.0,
            strategy_type="baseload"
        )

        assert result["annual_operating_hours"] > 0
        assert result["annual_electrical_generation_kwh"] > 0
        assert result["annual_thermal_generation_mmbtu"] > 0

    def test_hourly_dispatch_schedule(self, agent):
        """Test hourly dispatch schedule is provided"""
        result = agent.optimize_operating_strategy(
            electrical_load_profile_kw=[2000] * 24,
            thermal_load_profile_mmbtu_hr=[15.0] * 24,
            chp_electrical_capacity_kw=2000,
            chp_thermal_capacity_mmbtu_hr=15.0,
            electricity_rate_schedule=[0.12] * 24,
            gas_price_per_mmbtu=6.0,
            strategy_type="thermal_following"
        )

        assert "hourly_dispatch_schedule" in result
        assert len(result["hourly_dispatch_schedule"]) > 0


# ============================================================================
# CATEGORY 8: TOOL 7 - CALCULATE EMISSIONS REDUCTION (6 tests)
# ============================================================================

class TestCalculateEmissionsReduction:
    """Test emissions reduction calculation tool"""

    def test_emissions_reduction_calculation(self, agent):
        """Test basic emissions reduction calculation"""
        result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80
        )

        assert result["emissions_reduction_tonnes_co2_annual"] > 0
        assert result["emissions_reduction_percent"] > 0
        assert result["deterministic"] is True

    def test_chp_emissions_calculation(self, agent):
        """Test CHP system emissions calculation"""
        result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80,
            include_upstream_emissions=True
        )

        assert result["chp_total_emissions_tonnes_co2"] > 0
        assert result["chp_combustion_emissions_tonnes_co2"] > 0
        assert result["chp_upstream_emissions_tonnes_co2"] > 0

    def test_baseline_emissions_calculation(self, agent):
        """Test baseline emissions calculation"""
        result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80
        )

        assert result["baseline_total_emissions_tonnes_co2"] > 0
        assert result["baseline_electricity_emissions_tonnes_co2"] > 0
        assert result["baseline_thermal_emissions_tonnes_co2"] > 0

    def test_emission_intensity_comparison(self, agent):
        """Test emission intensity comparison"""
        result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80
        )

        # CHP should have lower emission intensity than baseline
        assert result["chp_emission_intensity_kg_co2_per_kwh_equivalent"] < result["baseline_emission_intensity_kg_co2_per_kwh_equivalent"]

    def test_biogas_zero_emissions(self, agent):
        """Test biogas CHP has zero carbon emissions"""
        result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=200_000,  # More fuel for biogas
            chp_fuel_type="biogas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80,
            include_upstream_emissions=False
        )

        # Biogas CHP should have zero emissions
        assert result["chp_combustion_emissions_tonnes_co2"] == 0

    def test_upstream_emissions_optional(self, agent):
        """Test upstream emissions can be excluded"""
        result_with = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80,
            include_upstream_emissions=True
        )

        result_without = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80,
            include_upstream_emissions=False
        )

        assert result_with["chp_upstream_emissions_tonnes_co2"] > 0
        assert result_without["chp_upstream_emissions_tonnes_co2"] == 0


# ============================================================================
# CATEGORY 9: TOOL 8 - GENERATE CHP REPORT (4 tests)
# ============================================================================

class TestGenerateCHPReport:
    """Test CHP report generation tool"""

    def test_comprehensive_report_generation(self, agent):
        """Test comprehensive report generation"""
        # First get results from other tools
        tech_result = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )

        perf_result = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )

        econ_result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        emis_result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=16_000_000,
            chp_thermal_output_mmbtu_annual=120_000,
            chp_fuel_input_mmbtu_annual=144_000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80
        )

        report = agent.generate_chp_report(
            technology_selection_result=tech_result,
            performance_result=perf_result,
            economic_result=econ_result,
            emissions_result=emis_result,
            facility_name="ABC Manufacturing"
        )

        assert report["facility_name"] == "ABC Manufacturing"
        assert "executive_summary" in report
        assert "technical_summary" in report
        assert "economic_summary" in report
        assert "environmental_summary" in report
        assert report["deterministic"] is True

    def test_report_recommendations(self, agent):
        """Test report includes recommendations"""
        tech_result = {"recommended_technology": "reciprocating_engine", "technology_description": "Reciprocating Engine"}
        perf_result = {"electrical_efficiency": 0.38, "total_efficiency": 0.82, "thermal_efficiency": 0.44, "heat_recovery_effectiveness": 75}
        econ_result = {"simple_payback_years": 4.5, "npv_20yr": 2_000_000, "net_annual_savings": 800_000, "chp_capex_gross": 3_500_000, "net_capex_after_incentives": 3_000_000, "irr_percent": 18, "benefit_cost_ratio": 2.1}
        emis_result = {"emissions_reduction_tonnes_co2_annual": 3000, "emissions_reduction_percent": 35, "chp_emission_intensity_kg_co2_per_kwh_equivalent": 0.25, "baseline_emission_intensity_kg_co2_per_kwh_equivalent": 0.38}

        report = agent.generate_chp_report(
            technology_selection_result=tech_result,
            performance_result=perf_result,
            economic_result=econ_result,
            emissions_result=emis_result,
            facility_name="Test Facility"
        )

        assert "recommendations" in report
        assert len(report["recommendations"]) > 0

    def test_report_overall_recommendation(self, agent):
        """Test report provides overall recommendation"""
        tech_result = {"recommended_technology": "reciprocating_engine", "technology_description": "Reciprocating Engine"}
        perf_result = {"electrical_efficiency": 0.38, "total_efficiency": 0.82, "thermal_efficiency": 0.44, "heat_recovery_effectiveness": 75}
        econ_result = {"simple_payback_years": 4.2, "npv_20yr": 2_500_000, "net_annual_savings": 850_000, "chp_capex_gross": 3_500_000, "net_capex_after_incentives": 3_000_000, "irr_percent": 19, "benefit_cost_ratio": 2.3}
        emis_result = {"emissions_reduction_tonnes_co2_annual": 3000, "emissions_reduction_percent": 35, "chp_emission_intensity_kg_co2_per_kwh_equivalent": 0.25, "baseline_emission_intensity_kg_co2_per_kwh_equivalent": 0.38}

        report = agent.generate_chp_report(
            technology_selection_result=tech_result,
            performance_result=perf_result,
            economic_result=econ_result,
            emissions_result=emis_result
        )

        assert report["overall_recommendation"] == "PROCEED"

    def test_report_with_optional_results(self, agent):
        """Test report handles optional interconnection and strategy results"""
        tech_result = {"recommended_technology": "reciprocating_engine", "technology_description": "Reciprocating Engine"}
        perf_result = {"electrical_efficiency": 0.38, "total_efficiency": 0.82, "thermal_efficiency": 0.44, "heat_recovery_effectiveness": 75}
        econ_result = {"simple_payback_years": 4.5, "npv_20yr": 2_000_000, "net_annual_savings": 800_000, "chp_capex_gross": 3_500_000, "net_capex_after_incentives": 3_000_000, "irr_percent": 18, "benefit_cost_ratio": 2.1}
        emis_result = {"emissions_reduction_tonnes_co2_annual": 3000, "emissions_reduction_percent": 35, "chp_emission_intensity_kg_co2_per_kwh_equivalent": 0.25, "baseline_emission_intensity_kg_co2_per_kwh_equivalent": 0.38}
        intercon_result = {"utility_application_timeline_weeks": 8}
        strategy_result = {"recommended_strategy": "thermal_following"}

        report = agent.generate_chp_report(
            technology_selection_result=tech_result,
            performance_result=perf_result,
            economic_result=econ_result,
            emissions_result=emis_result,
            interconnection_result=intercon_result,
            operating_strategy_result=strategy_result
        )

        # Check that optional results are incorporated
        assert any("8 weeks" in rec for rec in report["recommendations"])
        assert any("thermal_following" in rec for rec in report["recommendations"])


# ============================================================================
# CATEGORY 10: INTEGRATION TESTS (3 tests)
# ============================================================================

class TestIntegration:
    """Test integration across multiple tools"""

    def test_full_analysis_workflow(self, agent):
        """Test complete analysis workflow using all tools"""
        # Step 1: Select technology
        tech_result = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )
        assert tech_result["recommended_technology"] in ["reciprocating_engine", "gas_turbine"]

        # Step 2: Calculate performance
        perf_result = agent.calculate_chp_performance(
            chp_technology=tech_result["recommended_technology"],
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )
        assert perf_result["total_efficiency"] > 0.70

        # Step 3: Calculate economics
        econ_result = agent.calculate_economic_metrics(
            electrical_output_kw=perf_result["electrical_output_kw"],
            thermal_output_mmbtu_hr=perf_result["thermal_output_mmbtu_hr"],
            fuel_input_mmbtu_hr=perf_result["fuel_input_mmbtu_hr"],
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )
        assert econ_result["simple_payback_years"] < 10

        # Step 4: Calculate emissions
        emis_result = agent.calculate_emissions_reduction(
            chp_electrical_output_kwh_annual=econ_result["annual_kwh_generated"],
            chp_thermal_output_mmbtu_annual=econ_result["annual_thermal_mmbtu_generated"],
            chp_fuel_input_mmbtu_annual=perf_result["fuel_input_mmbtu_hr"] * 8000,
            chp_fuel_type="natural_gas",
            baseline_grid_emissions_kg_co2_per_kwh=0.45,
            baseline_thermal_fuel_type="natural_gas",
            baseline_boiler_efficiency=0.80
        )
        assert emis_result["emissions_reduction_tonnes_co2_annual"] > 0

        # Step 5: Generate report
        report = agent.generate_chp_report(
            technology_selection_result=tech_result,
            performance_result=perf_result,
            economic_result=econ_result,
            emissions_result=emis_result,
            facility_name="Integration Test Facility"
        )
        assert report["overall_recommendation"] in ["PROCEED", "CONSIDER"]

    def test_technology_comparison_workflow(self, agent):
        """Test comparing multiple technologies"""
        technologies = ["reciprocating_engine", "gas_turbine", "microturbine"]
        results = []

        for tech in technologies:
            try:
                result = agent.calculate_chp_performance(
                    chp_technology=tech,
                    electrical_capacity_kw=1000,
                    fuel_input_mmbtu_hr=10.0,
                    heat_recovery_configuration="jacket_exhaust",
                    exhaust_temperature_f=850,
                    exhaust_mass_flow_lb_hr=15000,
                    heat_recovery_target_temperature_f=250
                )
                results.append({
                    "technology": tech,
                    "electrical_efficiency": result["electrical_efficiency"],
                    "total_efficiency": result["total_efficiency"]
                })
            except:
                pass

        assert len(results) >= 3
        # Fuel cell should have highest electrical efficiency if included
        # Reciprocating engine should have good total efficiency

    def test_sensitivity_analysis_workflow(self, agent):
        """Test sensitivity analysis by varying key parameters"""
        base_result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        # Test sensitivity to gas price
        high_gas_result = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=9.0,  # 50% higher
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=9.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        # Higher gas price should worsen economics (longer payback, lower NPV)
        assert high_gas_result["simple_payback_years"] > base_result["simple_payback_years"]
        assert high_gas_result["npv_20yr"] < base_result["npv_20yr"]


# ============================================================================
# CATEGORY 11: DETERMINISM TESTS (3 tests)
# ============================================================================

class TestDeterminism:
    """Test deterministic behavior across runs"""

    def test_technology_selection_determinism(self, agent):
        """Test technology selection produces identical results"""
        result1 = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )

        result2 = agent.select_chp_technology(
            electrical_demand_kw=2000,
            thermal_demand_mmbtu_hr=15.0,
            heat_to_power_ratio=2.2,
            load_profile_type="baseload_24x7",
            available_fuels=["natural_gas"],
            emissions_requirements="low_nox",
            space_constraints="moderate"
        )

        assert result1["recommended_technology"] == result2["recommended_technology"]
        assert result1["typical_electrical_efficiency"] == result2["typical_electrical_efficiency"]
        assert result1["selection_score"] == result2["selection_score"]

    def test_performance_calculation_determinism(self, agent):
        """Test performance calculation produces identical results"""
        result1 = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )

        result2 = agent.calculate_chp_performance(
            chp_technology="reciprocating_engine",
            electrical_capacity_kw=2000,
            fuel_input_mmbtu_hr=18.0,
            heat_recovery_configuration="jacket_exhaust",
            exhaust_temperature_f=850,
            exhaust_mass_flow_lb_hr=25000,
            heat_recovery_target_temperature_f=250
        )

        assert result1["electrical_efficiency"] == result2["electrical_efficiency"]
        assert result1["thermal_output_mmbtu_hr"] == result2["thermal_output_mmbtu_hr"]
        assert result1["total_efficiency"] == result2["total_efficiency"]

    def test_economic_calculation_determinism(self, agent):
        """Test economic calculation produces identical results"""
        result1 = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        result2 = agent.calculate_economic_metrics(
            electrical_output_kw=2000,
            thermal_output_mmbtu_hr=15.0,
            fuel_input_mmbtu_hr=18.0,
            annual_operating_hours=8000,
            electricity_rate_per_kwh=0.12,
            demand_charge_per_kw_month=15.0,
            gas_price_per_mmbtu=6.0,
            thermal_fuel_displaced="natural_gas",
            thermal_fuel_price_per_mmbtu=6.0,
            thermal_boiler_efficiency=0.80,
            chp_capex_usd=3_500_000,
            chp_opex_per_kwh=0.015
        )

        assert result1["simple_payback_years"] == result2["simple_payback_years"]
        assert result1["npv_20yr"] == result2["npv_20yr"]
        assert result1["spark_spread_per_mwh"] == result2["spark_spread_per_mwh"]


# ============================================================================
# CATEGORY 12: ERROR HANDLING TESTS (6 tests)
# ============================================================================

class TestErrorHandling:
    """Test error handling and input validation"""

    def test_negative_electrical_demand_error(self, agent):
        """Test error on negative electrical demand"""
        with pytest.raises(ValueError, match="electrical_demand_kw must be positive"):
            agent.select_chp_technology(
                electrical_demand_kw=-1000,
                thermal_demand_mmbtu_hr=15.0,
                heat_to_power_ratio=2.2,
                load_profile_type="baseload_24x7",
                available_fuels=["natural_gas"],
                emissions_requirements="low_nox",
                space_constraints="moderate"
            )

    def test_invalid_load_profile_error(self, agent):
        """Test error on invalid load profile type"""
        with pytest.raises(ValueError, match="load_profile_type must be one of"):
            agent.select_chp_technology(
                electrical_demand_kw=2000,
                thermal_demand_mmbtu_hr=15.0,
                heat_to_power_ratio=2.2,
                load_profile_type="invalid_profile",
                available_fuels=["natural_gas"],
                emissions_requirements="low_nox",
                space_constraints="moderate"
            )

    def test_invalid_part_load_ratio_error(self, agent):
        """Test error on invalid part load ratio"""
        with pytest.raises(ValueError, match="part_load_ratio must be between"):
            agent.calculate_chp_performance(
                chp_technology="reciprocating_engine",
                electrical_capacity_kw=2000,
                fuel_input_mmbtu_hr=18.0,
                heat_recovery_configuration="jacket_exhaust",
                exhaust_temperature_f=850,
                exhaust_mass_flow_lb_hr=25000,
                heat_recovery_target_temperature_f=250,
                part_load_ratio=1.5  # Invalid: > 1.0
            )

    def test_invalid_operating_hours_error(self, agent):
        """Test error on invalid annual operating hours"""
        with pytest.raises(ValueError, match="annual_operating_hours must be between"):
            agent.calculate_economic_metrics(
                electrical_output_kw=2000,
                thermal_output_mmbtu_hr=15.0,
                fuel_input_mmbtu_hr=18.0,
                annual_operating_hours=9000,  # Invalid: > 8760
                electricity_rate_per_kwh=0.12,
                demand_charge_per_kw_month=15.0,
                gas_price_per_mmbtu=6.0,
                thermal_fuel_displaced="natural_gas",
                thermal_fuel_price_per_mmbtu=6.0,
                thermal_boiler_efficiency=0.80,
                chp_capex_usd=3_500_000,
                chp_opex_per_kwh=0.015
            )

    def test_invalid_voltage_level_error(self, agent):
        """Test error on invalid voltage level"""
        with pytest.raises(ValueError, match="voltage_level must be one of"):
            agent.assess_grid_interconnection(
                chp_electrical_capacity_kw=2000,
                facility_peak_demand_kw=3500,
                voltage_level="invalid_voltage",
                export_mode="no_export",
                utility_territory="investor_owned"
            )

    def test_invalid_strategy_type_error(self, agent):
        """Test error on invalid operating strategy type"""
        with pytest.raises(ValueError, match="strategy_type must be one of"):
            agent.optimize_operating_strategy(
                electrical_load_profile_kw=[2000] * 24,
                thermal_load_profile_mmbtu_hr=[15.0] * 24,
                chp_electrical_capacity_kw=2000,
                chp_thermal_capacity_mmbtu_hr=15.0,
                electricity_rate_schedule=[0.12] * 24,
                gas_price_per_mmbtu=6.0,
                strategy_type="invalid_strategy"
            )


# ============================================================================
# HELPER TESTS
# ============================================================================

class TestHelperMethods:
    """Test helper methods"""

    def test_health_check(self, agent):
        """Test health check returns healthy status"""
        result = agent.health_check()
        assert result["status"] == "healthy"
        assert result["agent"] == "TestCogenerationCHPAgent"
        assert "version" in result

    def test_ready_check(self, agent):
        """Test readiness check"""
        result = agent.ready_check()
        assert result["status"] in ["ready", "not_ready"]
        if result["status"] == "ready":
            assert "dependencies" in result


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
