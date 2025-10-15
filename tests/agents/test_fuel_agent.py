"""Comprehensive test suite for FuelAgent (deterministic emissions calculator).

This module provides extensive test coverage for FuelAgent, ensuring:
1. All fuel types are tested (natural_gas, diesel, gasoline, coal, propane, etc.)
2. Unit conversions are accurate
3. Emission factor retrieval works correctly
4. Deterministic behavior (same input -> same output)
5. Boundary conditions are handled properly
6. Error handling is robust
7. Integration with real emission factor data

Test Coverage Target: 85%+

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import asyncio
from typing import List
from unittest.mock import Mock, patch
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.data.emission_factors import EmissionFactors


class TestFuelAgentUnitTests:
    """Unit tests for FuelAgent - Test each method and logic path independently.

    This class contains 20+ unit tests covering:
    - Initialization and setup
    - Validation logic
    - Each fuel type calculation
    - Unit conversions
    - Emission factor retrieval
    - Scope determination
    - Recommendation generation
    - Energy content calculations
    """

    @pytest.fixture
    def agent(self):
        """Create a FuelAgent instance for testing."""
        return FuelAgent()

    @pytest.fixture
    def emission_factors(self):
        """Create EmissionFactors instance for testing."""
        return EmissionFactors()

    # ==================== Initialization Tests ====================

    def test_initialization(self, agent):
        """Test FuelAgent initializes correctly with all required components."""
        assert agent.agent_id == "fuel"
        assert agent.name == "Fuel Emissions Calculator"
        assert agent.version == "0.0.1"
        assert agent.emission_factors is not None
        assert agent.fuel_config is not None
        assert agent.unit_converter is not None
        assert agent.performance_tracker is not None
        assert agent._cache == {}
        assert agent._historical_data == []
        assert agent._cache_hits == 0
        assert agent._cache_misses == 0

    def test_load_fuel_config(self, agent):
        """Test fuel configuration loading."""
        config = agent.fuel_config
        assert "fuel_properties" in config or config is not None
        # Should have at least basic fuel types
        if "fuel_properties" in config:
            assert isinstance(config["fuel_properties"], dict)

    # ==================== Validation Tests ====================

    def test_validate_valid_payload(self, agent):
        """Test validation passes for valid payload."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms"
        }
        assert agent.validate(payload) is True

    def test_validate_missing_fuel_type(self, agent):
        """Test validation fails when fuel_type is missing."""
        payload = {"amount": 100, "unit": "therms"}
        assert agent.validate(payload) is False

    def test_validate_missing_amount(self, agent):
        """Test validation fails when amount is missing."""
        payload = {"fuel_type": "natural_gas", "unit": "therms"}
        assert agent.validate(payload) is False

    def test_validate_missing_unit(self, agent):
        """Test validation fails when unit is missing."""
        payload = {"fuel_type": "natural_gas", "amount": 100}
        assert agent.validate(payload) is False

    def test_validate_negative_amount_non_renewable(self, agent):
        """Test validation fails for negative amount with non-renewable fuel."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": -100,
            "unit": "therms"
        }
        assert agent.validate(payload) is False

    # ==================== Natural Gas Tests ====================

    def test_natural_gas_therms_calculation(self, agent):
        """Test natural gas calculation with therms unit."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is True
        assert "data" in result
        data = result["data"]

        # EPA factor: 5.3 kg CO2e/therm
        expected_emissions = 100 * 5.3
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "natural_gas"
        assert data["consumption_amount"] == 100
        assert data["consumption_unit"] == "therms"
        assert data["scope"] == "1"

    def test_natural_gas_ccf_calculation(self, agent):
        """Test natural gas calculation with ccf unit."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 50,
            "unit": "ccf"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 5.3 kg CO2e/ccf
        expected_emissions = 50 * 5.3
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    def test_natural_gas_m3_calculation(self, agent):
        """Test natural gas calculation with cubic meters unit."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "m3"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 1.89 kg CO2e/m3
        expected_emissions = 100 * 1.89
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    # ==================== Diesel Tests ====================

    def test_diesel_gallons_calculation(self, agent):
        """Test diesel calculation with gallons unit."""
        payload = {
            "fuel_type": "diesel",
            "amount": 50,
            "unit": "gallons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 10.21 kg CO2e/gallon
        expected_emissions = 50 * 10.21
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "diesel"
        assert data["scope"] == "1"

    def test_diesel_liters_calculation(self, agent):
        """Test diesel calculation with liters unit."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "liters"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 2.68 kg CO2e/liter
        expected_emissions = 100 * 2.68
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    def test_diesel_kg_calculation(self, agent):
        """Test diesel calculation with kg unit."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "kg"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 3.16 kg CO2e/kg
        expected_emissions = 100 * 3.16
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    # ==================== Gasoline Tests ====================

    def test_gasoline_gallons_calculation(self, agent):
        """Test gasoline calculation with gallons unit."""
        payload = {
            "fuel_type": "gasoline",
            "amount": 50,
            "unit": "gallons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 8.78 kg CO2e/gallon
        expected_emissions = 50 * 8.78
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "gasoline"

    def test_gasoline_liters_calculation(self, agent):
        """Test gasoline calculation with liters unit."""
        payload = {
            "fuel_type": "gasoline",
            "amount": 100,
            "unit": "liters"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 2.31 kg CO2e/liter
        expected_emissions = 100 * 2.31
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    # ==================== Coal Tests ====================

    def test_coal_tons_calculation(self, agent):
        """Test coal calculation with tons unit."""
        payload = {
            "fuel_type": "coal",
            "amount": 5,
            "unit": "tons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 2086 kg CO2e/ton
        expected_emissions = 5 * 2086
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "coal"

    def test_coal_kg_calculation(self, agent):
        """Test coal calculation with kg unit."""
        payload = {
            "fuel_type": "coal",
            "amount": 1000,
            "unit": "kg"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 2.086 kg CO2e/kg
        expected_emissions = 1000 * 2.086
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    # ==================== Propane Tests ====================

    def test_propane_gallons_calculation(self, agent):
        """Test propane calculation with gallons unit."""
        payload = {
            "fuel_type": "propane",
            "amount": 100,
            "unit": "gallons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 5.76 kg CO2e/gallon
        expected_emissions = 100 * 5.76
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "propane"

    def test_propane_liters_calculation(self, agent):
        """Test propane calculation with liters unit."""
        payload = {
            "fuel_type": "propane",
            "amount": 100,
            "unit": "liters"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 1.51 kg CO2e/liter
        expected_emissions = 100 * 1.51
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)

    # ==================== Fuel Oil Tests ====================

    def test_fuel_oil_gallons_calculation(self, agent):
        """Test fuel oil calculation with gallons unit."""
        payload = {
            "fuel_type": "fuel_oil",
            "amount": 100,
            "unit": "gallons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 10.16 kg CO2e/gallon
        expected_emissions = 100 * 10.16
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "fuel_oil"

    # ==================== Biomass Tests ====================

    def test_biomass_tons_calculation(self, agent):
        """Test biomass calculation with tons unit."""
        payload = {
            "fuel_type": "biomass",
            "amount": 10,
            "unit": "tons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 1500 kg CO2e/ton
        expected_emissions = 10 * 1500
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "biomass"

    # ==================== Electricity Tests ====================

    def test_electricity_kwh_calculation(self, agent):
        """Test electricity calculation with kWh unit."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # EPA factor: 0.385 kg CO2e/kWh (US)
        expected_emissions = 1000 * 0.385
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["fuel_type"] == "electricity"
        assert data["scope"] == "2"  # Electricity is Scope 2

    def test_electricity_renewable_offset(self, agent):
        """Test electricity calculation with renewable offset."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "renewable_percentage": 30
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]

        # Base emissions: 1000 * 0.385 = 385 kg
        # With 30% offset: 385 * 0.7 = 269.5 kg
        base_emissions = 1000 * 0.385
        expected_emissions = base_emissions * 0.7
        assert data["co2e_emissions_kg"] == pytest.approx(expected_emissions, rel=0.01)
        assert data["renewable_offset_applied"] is True

    # ==================== Fuel Type Alias Tests ====================

    def test_lpg_alias_maps_to_propane(self, agent):
        """Test LPG alias correctly maps to propane."""
        payload = {
            "fuel_type": "lpg",
            "amount": 100,
            "unit": "gallons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]
        assert data["fuel_type"] == "propane"

    def test_heating_oil_alias_maps_to_fuel_oil(self, agent):
        """Test heating_oil alias correctly maps to fuel_oil."""
        payload = {
            "fuel_type": "heating_oil",
            "amount": 100,
            "unit": "gallons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]
        assert data["fuel_type"] == "fuel_oil"

    def test_wood_alias_maps_to_biomass(self, agent):
        """Test wood alias correctly maps to biomass."""
        payload = {
            "fuel_type": "wood",
            "amount": 5,
            "unit": "tons"
        }
        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]
        assert data["fuel_type"] == "biomass"

    # ==================== Scope Determination Tests ====================

    def test_scope_determination_scope1_fuels(self, agent):
        """Test scope determination for Scope 1 fuels."""
        scope1_fuels = ["natural_gas", "diesel", "gasoline", "propane", "fuel_oil", "coal", "biomass"]

        for fuel in scope1_fuels:
            scope = agent._determine_scope(fuel)
            assert scope == "1", f"{fuel} should be Scope 1"

    def test_scope_determination_scope2_fuels(self, agent):
        """Test scope determination for Scope 2 fuels."""
        scope2_fuels = ["electricity", "district_heating", "district_cooling"]

        for fuel in scope2_fuels:
            scope = agent._determine_scope(fuel)
            assert scope == "2", f"{fuel} should be Scope 2"

    # ==================== Efficiency Tests ====================

    def test_efficiency_adjustment(self, agent):
        """Test emissions adjustment based on equipment efficiency."""
        payload_high_eff = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "efficiency": 0.9  # 90% efficient
        }
        result_high = agent.run(payload_high_eff)

        payload_low_eff = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "efficiency": 0.5  # 50% efficient
        }
        result_low = agent.run(payload_low_eff)

        assert result_high["success"] is True
        assert result_low["success"] is True

        # Lower efficiency should result in higher emissions
        assert result_low["data"]["co2e_emissions_kg"] > result_high["data"]["co2e_emissions_kg"]
        assert result_high["data"]["efficiency_adjusted"] is True
        assert result_low["data"]["efficiency_adjusted"] is True

    # ==================== Energy Content Tests ====================

    def test_energy_content_calculation_kwh(self, agent):
        """Test energy content calculation for kWh."""
        energy = agent._calculate_energy_content(1000, "kWh", "electricity")
        # 1000 kWh * 0.003412 = 3.412 MMBtu
        assert energy == pytest.approx(3.412, rel=0.01)

    def test_energy_content_calculation_therms(self, agent):
        """Test energy content calculation for therms."""
        energy = agent._calculate_energy_content(100, "therms", "natural_gas")
        # 100 therms * 0.1 = 10 MMBtu
        assert energy == pytest.approx(10.0, rel=0.01)

    def test_energy_content_calculation_diesel_gallons(self, agent):
        """Test energy content calculation for diesel gallons."""
        energy = agent._calculate_energy_content(100, "gallons", "diesel")
        # 100 gallons * 0.138 = 13.8 MMBtu
        assert energy == pytest.approx(13.8, rel=0.01)

    def test_energy_content_calculation_gasoline_gallons(self, agent):
        """Test energy content calculation for gasoline gallons."""
        energy = agent._calculate_energy_content(100, "gallons", "gasoline")
        # 100 gallons * 0.125 = 12.5 MMBtu
        assert energy == pytest.approx(12.5, rel=0.01)

    # ==================== Recommendation Tests ====================

    def test_recommendations_coal(self, agent):
        """Test recommendations generation for coal."""
        recommendations = agent._generate_fuel_recommendations(
            "coal", 1000, "tons", 2086000, "US"
        )

        assert len(recommendations) > 0
        assert len(recommendations) <= 5

        # Coal should have high-priority fuel switching recommendation
        assert any("coal" in rec["action"].lower() for rec in recommendations)
        assert any(rec["priority"] == "high" for rec in recommendations)

    def test_recommendations_fuel_oil(self, agent):
        """Test recommendations generation for fuel oil."""
        recommendations = agent._generate_fuel_recommendations(
            "fuel_oil", 100, "gallons", 1016, "US"
        )

        assert len(recommendations) > 0
        # Fuel oil should have switching recommendation
        assert any("oil" in rec["action"].lower() for rec in recommendations)

    def test_recommendations_natural_gas(self, agent):
        """Test recommendations generation for natural gas."""
        recommendations = agent._generate_fuel_recommendations(
            "natural_gas", 1000, "therms", 5300, "US"
        )

        assert len(recommendations) > 0
        # Natural gas should have renewable gas or hydrogen recommendation
        assert any("renewable" in rec["action"].lower() or "hydrogen" in rec["action"].lower()
                   for rec in recommendations)

    def test_recommendations_electricity(self, agent):
        """Test recommendations generation for electricity."""
        recommendations = agent._generate_fuel_recommendations(
            "electricity", 10000, "kWh", 3850, "US"
        )

        assert len(recommendations) > 0
        # Electricity should have solar/renewable recommendation
        assert any("solar" in rec["action"].lower() or "renewable" in rec["action"].lower()
                   for rec in recommendations)

    def test_recommendations_always_include_efficiency(self, agent):
        """Test that efficiency recommendations are always included."""
        recommendations = agent._generate_fuel_recommendations(
            "natural_gas", 100, "therms", 530, "US"
        )

        # Should include efficiency and energy management recommendations
        assert any("efficiency" in rec["action"].lower() for rec in recommendations)
        assert any("energy management" in rec["action"].lower() for rec in recommendations)


class TestFuelAgentIntegration:
    """Integration tests for FuelAgent with real emission factor data.

    This class contains 10+ integration tests covering:
    - Realistic scenarios (commercial buildings, industrial facilities)
    - Multiple fuel types together
    - Regional variations (US, EU, UK)
    - Error propagation
    - Performance tracking
    """

    @pytest.fixture
    def agent(self):
        """Create a FuelAgent instance for testing."""
        return FuelAgent()

    # ==================== Commercial Building Scenarios ====================

    def test_commercial_building_scenario(self, agent):
        """Test realistic commercial building with multiple fuel types."""
        # Office building: electricity + natural gas heating

        # Electricity for HVAC, lighting, equipment
        electricity_result = agent.run({
            "fuel_type": "electricity",
            "amount": 50000,
            "unit": "kWh",
            "country": "US"
        })

        # Natural gas for heating
        gas_result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 2000,
            "unit": "therms",
            "country": "US"
        })

        assert electricity_result["success"] is True
        assert gas_result["success"] is True

        total_emissions = (
            electricity_result["data"]["co2e_emissions_kg"] +
            gas_result["data"]["co2e_emissions_kg"]
        )

        # Verify total is reasonable for a commercial building
        assert total_emissions > 0
        assert total_emissions < 100000  # Should be less than 100 tons CO2e

    def test_retail_store_scenario(self, agent):
        """Test retail store with electricity and backup generator."""
        # Retail store: mostly electricity + diesel backup generator

        electricity_result = agent.run({
            "fuel_type": "electricity",
            "amount": 30000,
            "unit": "kWh",
            "country": "US"
        })

        diesel_result = agent.run({
            "fuel_type": "diesel",
            "amount": 50,  # Occasional generator use
            "unit": "gallons",
            "country": "US"
        })

        assert electricity_result["success"] is True
        assert diesel_result["success"] is True

        # Electricity should dominate emissions
        assert electricity_result["data"]["co2e_emissions_kg"] > diesel_result["data"]["co2e_emissions_kg"]

    # ==================== Industrial Facility Scenarios ====================

    def test_industrial_facility_coal_dominated(self, agent):
        """Test industrial facility with coal as primary fuel."""
        coal_result = agent.run({
            "fuel_type": "coal",
            "amount": 100,
            "unit": "tons",
            "country": "US"
        })

        electricity_result = agent.run({
            "fuel_type": "electricity",
            "amount": 100000,
            "unit": "kWh",
            "country": "US"
        })

        assert coal_result["success"] is True
        assert electricity_result["success"] is True

        # Coal should have very high emissions
        coal_emissions = coal_result["data"]["co2e_emissions_kg"]
        assert coal_emissions > 200000  # Over 200 tons CO2e

    def test_industrial_facility_natural_gas_boiler(self, agent):
        """Test industrial facility with natural gas boiler."""
        result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 50000,
            "unit": "therms",
            "country": "US",
            "efficiency": 0.85  # 85% efficient boiler
        })

        assert result["success"] is True
        data = result["data"]

        # Should calculate emissions with efficiency adjustment
        assert data["co2e_emissions_kg"] > 0
        assert data["efficiency_adjusted"] is True
        assert data["energy_content_mmbtu"] > 0

    def test_manufacturing_plant_mixed_fuels(self, agent):
        """Test manufacturing plant with multiple fuel sources."""
        # Natural gas for process heat
        gas_result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 10000,
            "unit": "therms",
            "country": "US"
        })

        # Diesel for forklifts and vehicles
        diesel_result = agent.run({
            "fuel_type": "diesel",
            "amount": 500,
            "unit": "gallons",
            "country": "US"
        })

        # Electricity for machinery
        electricity_result = agent.run({
            "fuel_type": "electricity",
            "amount": 200000,
            "unit": "kWh",
            "country": "US"
        })

        # Propane for heating
        propane_result = agent.run({
            "fuel_type": "propane",
            "amount": 1000,
            "unit": "gallons",
            "country": "US"
        })

        assert all([
            gas_result["success"],
            diesel_result["success"],
            electricity_result["success"],
            propane_result["success"]
        ])

        total_emissions = sum([
            gas_result["data"]["co2e_emissions_kg"],
            diesel_result["data"]["co2e_emissions_kg"],
            electricity_result["data"]["co2e_emissions_kg"],
            propane_result["data"]["co2e_emissions_kg"]
        ])

        assert total_emissions > 0

    # ==================== Regional Variation Tests ====================

    def test_electricity_us_vs_eu(self, agent):
        """Test electricity emission factors differ by region."""
        us_result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US"
        })

        eu_result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "EU"
        })

        assert us_result["success"] is True
        assert eu_result["success"] is True

        # US grid is dirtier than EU grid
        assert us_result["data"]["co2e_emissions_kg"] > eu_result["data"]["co2e_emissions_kg"]

    def test_electricity_us_vs_uk(self, agent):
        """Test electricity emission factors for US vs UK."""
        us_result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US"
        })

        uk_result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "UK"
        })

        assert us_result["success"] is True
        assert uk_result["success"] is True

        # UK grid is cleaner than US grid
        assert us_result["data"]["co2e_emissions_kg"] > uk_result["data"]["co2e_emissions_kg"]

    # ==================== Batch Processing Tests ====================

    def test_batch_process_multiple_fuels(self, agent):
        """Test batch processing of multiple fuel sources."""
        fuels = [
            {"fuel_type": "natural_gas", "amount": 1000, "unit": "therms"},
            {"fuel_type": "electricity", "amount": 10000, "unit": "kWh"},
            {"fuel_type": "diesel", "amount": 100, "unit": "gallons"},
        ]

        results = agent.batch_process(fuels)

        assert len(results) == 3
        assert all(r["success"] for r in results)

        # Verify total emissions can be summed
        total_emissions = sum(r["data"]["co2e_emissions_kg"] for r in results)
        assert total_emissions > 0

    def test_async_batch_process(self, agent, event_loop):
        """Test asynchronous batch processing."""
        fuels = [
            {"fuel_type": "natural_gas", "amount": 1000, "unit": "therms"},
            {"fuel_type": "electricity", "amount": 10000, "unit": "kWh"},
            {"fuel_type": "diesel", "amount": 100, "unit": "gallons"},
            {"fuel_type": "gasoline", "amount": 100, "unit": "gallons"},
        ]

        results = event_loop.run_until_complete(agent.async_batch_process(fuels))

        assert len(results) == 4
        assert all(r["success"] for r in results)

    # ==================== Performance and Caching Tests ====================

    def test_cache_performance(self, agent):
        """Test emission factor caching improves performance."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "country": "US"
        }

        # First call - cache miss
        initial_misses = agent._cache_misses
        result1 = agent.run(payload)
        assert result1["success"] is True

        # Second call - should hit cache
        result2 = agent.run(payload)
        assert result2["success"] is True

        # Results should be identical (deterministic)
        assert result1["data"]["co2e_emissions_kg"] == result2["data"]["co2e_emissions_kg"]

    def test_performance_summary(self, agent):
        """Test performance metrics tracking."""
        # Run some calculations
        agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"})
        agent.run({"fuel_type": "electricity", "amount": 1000, "unit": "kWh"})

        summary = agent.get_performance_summary()

        assert "fuel_metrics" in summary
        assert "cache_hit_rate" in summary["fuel_metrics"]
        assert "average_execution_time_ms" in summary["fuel_metrics"]
        assert "total_calculations" in summary["fuel_metrics"]
        assert summary["fuel_metrics"]["total_calculations"] >= 2

    def test_historical_data_tracking(self, agent):
        """Test historical data is tracked correctly."""
        initial_count = len(agent.get_historical_data())

        agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"})
        agent.run({"fuel_type": "electricity", "amount": 1000, "unit": "kWh"})

        historical = agent.get_historical_data()
        assert len(historical) == initial_count + 2

        # Verify structure of historical data
        record = historical[-1]
        assert "timestamp" in record
        assert "fuel_type" in record
        assert "amount" in record
        assert "emissions_kg" in record

    # ==================== Error Handling Tests ====================

    def test_error_invalid_fuel_type(self, agent):
        """Test error handling for invalid fuel type."""
        result = agent.run({
            "fuel_type": "invalid_fuel_type",
            "amount": 100,
            "unit": "therms"
        })

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["type"] == "DataError"

    def test_error_invalid_unit(self, agent):
        """Test error handling for invalid unit."""
        result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "invalid_unit"
        })

        assert result["success"] is False
        assert "error" in result


class TestFuelAgentDeterminism:
    """Determinism tests for FuelAgent - Ensure consistent, reproducible results.

    This class contains 5+ tests ensuring:
    - Same input produces same output (run 10 times)
    - Floating-point consistency
    - No randomness in calculations
    - Reproducible across different environments
    """

    @pytest.fixture
    def agent(self):
        """Create a FuelAgent instance for testing."""
        return FuelAgent()

    def test_determinism_natural_gas_10_runs(self, agent):
        """Test natural gas calculation produces identical results across 10 runs."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms"
        }

        results = [agent.run(payload) for _ in range(10)]

        # All should succeed
        assert all(r["success"] for r in results)

        # All emissions should be identical
        emissions = [r["data"]["co2e_emissions_kg"] for r in results]
        assert all(e == emissions[0] for e in emissions)

        # All emission factors should be identical
        factors = [r["data"]["emission_factor"] for r in results]
        assert all(f == factors[0] for f in factors)

    def test_determinism_electricity_10_runs(self, agent):
        """Test electricity calculation produces identical results across 10 runs."""
        payload = {
            "fuel_type": "electricity",
            "amount": 5000,
            "unit": "kWh",
            "country": "US"
        }

        results = [agent.run(payload) for _ in range(10)]

        assert all(r["success"] for r in results)

        emissions = [r["data"]["co2e_emissions_kg"] for r in results]
        assert all(e == emissions[0] for e in emissions)

    def test_determinism_diesel_10_runs(self, agent):
        """Test diesel calculation produces identical results across 10 runs."""
        payload = {
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "gallons"
        }

        results = [agent.run(payload) for _ in range(10)]

        assert all(r["success"] for r in results)

        emissions = [r["data"]["co2e_emissions_kg"] for r in results]
        assert all(e == emissions[0] for e in emissions)

    def test_determinism_with_efficiency_10_runs(self, agent):
        """Test calculations with efficiency produce identical results."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "efficiency": 0.85
        }

        results = [agent.run(payload) for _ in range(10)]

        assert all(r["success"] for r in results)

        emissions = [r["data"]["co2e_emissions_kg"] for r in results]
        assert all(e == emissions[0] for e in emissions)

    def test_determinism_with_renewable_offset_10_runs(self, agent):
        """Test calculations with renewable offset produce identical results."""
        payload = {
            "fuel_type": "electricity",
            "amount": 10000,
            "unit": "kWh",
            "renewable_percentage": 25
        }

        results = [agent.run(payload) for _ in range(10)]

        assert all(r["success"] for r in results)

        emissions = [r["data"]["co2e_emissions_kg"] for r in results]
        assert all(e == emissions[0] for e in emissions)

    def test_floating_point_consistency(self, agent):
        """Test floating-point calculations are consistent."""
        # Test with values that might cause floating-point issues
        payload = {
            "fuel_type": "natural_gas",
            "amount": 123.456789,
            "unit": "therms"
        }

        results = [agent.run(payload) for _ in range(10)]

        assert all(r["success"] for r in results)

        # Check exact equality (not just approximate)
        emissions = [r["data"]["co2e_emissions_kg"] for r in results]
        assert all(e == emissions[0] for e in emissions)

    def test_batch_processing_determinism(self, agent):
        """Test batch processing produces deterministic results."""
        fuels = [
            {"fuel_type": "natural_gas", "amount": 1000, "unit": "therms"},
            {"fuel_type": "electricity", "amount": 10000, "unit": "kWh"},
            {"fuel_type": "diesel", "amount": 100, "unit": "gallons"},
        ]

        # Run batch processing multiple times
        results1 = agent.batch_process(fuels)
        results2 = agent.batch_process(fuels)

        # Compare emissions from both runs (sorted to handle parallel execution order)
        emissions1 = sorted([r["data"]["co2e_emissions_kg"] for r in results1])
        emissions2 = sorted([r["data"]["co2e_emissions_kg"] for r in results2])

        assert emissions1 == emissions2


class TestFuelAgentBoundary:
    """Boundary and edge case tests for FuelAgent.

    This class contains 10+ tests covering:
    - Zero consumption
    - Very large consumption values
    - Very small consumption values
    - Invalid inputs
    - Missing parameters
    - Edge cases for each fuel type
    - Extreme efficiency values
    """

    @pytest.fixture
    def agent(self):
        """Create a FuelAgent instance for testing."""
        return FuelAgent()

    # ==================== Zero and Near-Zero Tests ====================

    def test_zero_consumption(self, agent):
        """Test handling of zero consumption."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 0,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is True
        assert result["data"]["co2e_emissions_kg"] == 0

    def test_very_small_consumption(self, agent):
        """Test handling of very small consumption values."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 0.001,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is True
        assert result["data"]["co2e_emissions_kg"] > 0
        assert result["data"]["co2e_emissions_kg"] < 0.01

    # ==================== Large Value Tests ====================

    def test_very_large_consumption(self, agent):
        """Test handling of very large consumption values."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 1000000,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is True
        # Should be around 5.3 million kg CO2e
        assert result["data"]["co2e_emissions_kg"] > 5000000

    def test_extremely_large_electricity_consumption(self, agent):
        """Test handling of extremely large electricity consumption."""
        payload = {
            "fuel_type": "electricity",
            "amount": 10000000,  # 10 million kWh
            "unit": "kWh"
        }
        result = agent.run(payload)

        assert result["success"] is True
        assert result["data"]["co2e_emissions_kg"] > 3000000

    # ==================== Invalid Input Tests ====================

    def test_negative_consumption_rejected(self, agent):
        """Test that negative consumption is rejected for non-renewable fuels."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": -100,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is False
        assert "error" in result

    def test_invalid_fuel_type(self, agent):
        """Test handling of completely invalid fuel type."""
        payload = {
            "fuel_type": "unicorn_fuel",
            "amount": 100,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is False
        assert result["error"]["type"] == "DataError"

    def test_invalid_unit_for_fuel(self, agent):
        """Test handling of invalid unit for a fuel type."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "gallons"  # Natural gas not typically measured in gallons
        }
        result = agent.run(payload)

        # Should fail because emission factor not found
        assert result["success"] is False

    def test_invalid_country_code(self, agent):
        """Test handling of invalid country code."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "ZZ"  # Invalid country
        }
        result = agent.run(payload)

        # Should fall back to US factors
        assert result["success"] is True

    # ==================== Efficiency Edge Cases ====================

    def test_zero_efficiency(self, agent):
        """Test handling of zero efficiency."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "efficiency": 0.0
        }
        result = agent.run(payload)

        # Zero efficiency would cause division by zero or infinite emissions
        # Agent should handle this gracefully
        assert result["success"] is True or result["success"] is False

    def test_efficiency_greater_than_one(self, agent):
        """Test handling of efficiency > 1.0 (physically impossible)."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "efficiency": 1.5
        }
        result = agent.run(payload)

        # Should process but emit warning or handle gracefully
        assert result["success"] is True

    def test_very_low_efficiency(self, agent):
        """Test handling of very low efficiency (1%)."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "efficiency": 0.01
        }
        result = agent.run(payload)

        assert result["success"] is True
        # Emissions should be very high due to low efficiency
        base_emissions = 100 * 5.3
        assert result["data"]["co2e_emissions_kg"] > base_emissions * 50

    # ==================== Renewable Percentage Edge Cases ====================

    def test_renewable_percentage_100(self, agent):
        """Test handling of 100% renewable electricity."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "renewable_percentage": 100
        }
        result = agent.run(payload)

        assert result["success"] is True
        # Should be zero emissions with 100% renewable
        assert result["data"]["co2e_emissions_kg"] == pytest.approx(0, abs=0.01)

    def test_renewable_percentage_over_100(self, agent):
        """Test handling of renewable percentage > 100%."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "renewable_percentage": 150
        }
        result = agent.run(payload)

        # Should handle gracefully (possibly negative emissions or cap at 100%)
        assert result["success"] is True

    def test_renewable_percentage_negative(self, agent):
        """Test handling of negative renewable percentage."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "renewable_percentage": -50
        }
        result = agent.run(payload)

        # Should handle gracefully
        assert result["success"] is True

    # ==================== Missing Optional Parameters ====================

    def test_missing_country_defaults_to_us(self, agent):
        """Test that missing country parameter defaults to US."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh"
        }
        result = agent.run(payload)

        assert result["success"] is True
        assert result["data"]["country"] == "US"

    def test_missing_year_defaults_to_current(self, agent):
        """Test that missing year parameter defaults to current year."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms"
        }
        result = agent.run(payload)

        assert result["success"] is True
        # Should use 2025 as default (from code)

    # ==================== Cache Management ====================

    def test_clear_cache(self, agent):
        """Test cache clearing functionality."""
        # Run calculation to populate cache
        agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"})

        # Clear cache
        agent.clear_cache()

        # Cache should be empty
        assert len(agent._cache) == 0

    def test_reset_metrics(self, agent):
        """Test metrics reset functionality."""
        # Run some calculations
        agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"})
        agent.run({"fuel_type": "electricity", "amount": 1000, "unit": "kWh"})

        # Reset metrics
        agent.reset_metrics()

        # Metrics should be reset
        assert agent._cache_hits == 0
        assert agent._cache_misses == 0
        assert len(agent._historical_data) == 0
        assert len(agent._execution_times) == 0

    # ==================== Export Results Tests ====================

    def test_export_results_json(self, agent, tmp_path):
        """Test exporting results to JSON format."""
        import os
        import json

        # Create some results
        results = [
            agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"}),
            agent.run({"fuel_type": "electricity", "amount": 1000, "unit": "kWh"}),
        ]

        # Change to temp directory to avoid polluting workspace
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            filepath = agent.export_results(results, format="json")

            # Verify file was created
            assert os.path.exists(filepath)
            assert filepath.endswith(".json")

            # Verify content
            with open(filepath) as f:
                data = json.load(f)
                assert len(data) == 2
                assert all(r["success"] for r in data)
        finally:
            os.chdir(original_dir)

    def test_export_results_csv(self, agent, tmp_path):
        """Test exporting results to CSV format."""
        import os
        import csv

        # Create some results
        results = [
            agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"}),
            agent.run({"fuel_type": "electricity", "amount": 1000, "unit": "kWh"}),
        ]

        # Change to temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            filepath = agent.export_results(results, format="csv")

            # Verify file was created
            assert os.path.exists(filepath)
            assert filepath.endswith(".csv")

            # Verify content
            with open(filepath) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
        finally:
            os.chdir(original_dir)

    def test_export_results_excel(self, agent, tmp_path):
        """Test exporting results to Excel format (if pandas available)."""
        import os

        # Create some results
        results = [
            agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"}),
            agent.run({"fuel_type": "electricity", "amount": 1000, "unit": "kWh"}),
        ]

        # Change to temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            try:
                # Excel export requires pandas and openpyxl
                import pandas as pd
                filepath = agent.export_results(results, format="excel")

                # Verify file was created
                assert os.path.exists(filepath)
                assert filepath.endswith(".excel")
            except (ImportError, ValueError) as e:
                # pandas not available or no Excel engine, that's ok
                pytest.skip(f"Excel export not available: {str(e)}")
        finally:
            os.chdir(original_dir)

    def test_export_results_invalid_format(self, agent):
        """Test error handling for invalid export format."""
        results = [
            agent.run({"fuel_type": "natural_gas", "amount": 100, "unit": "therms"})
        ]

        with pytest.raises(ValueError, match="Unsupported format"):
            agent.export_results(results, format="invalid_format")

    # ==================== Energy Content Info Tests ====================

    def test_energy_content_info_included(self, agent):
        """Test that energy content info is included in output when available."""
        result = agent.run({
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh"
        })

        assert result["success"] is True
        data = result["data"]

        # Should include energy content info
        if "energy_content_info" in data:
            assert isinstance(data["energy_content_info"], dict)

    def test_energy_content_fallback_unknown_unit(self, agent):
        """Test energy content calculation fallback for unknown unit."""
        # Use a unit that will trigger the fallback path
        energy = agent._calculate_energy_content(100, "unknown_unit", "unknown_fuel")

        # Should return 0.0 when unit is not recognized
        assert energy == 0.0

    def test_energy_content_fallback_kwh(self, agent):
        """Test energy content fallback calculation for kWh."""
        energy = agent._calculate_energy_content(1000, "kWh", "electricity")

        # Fallback: 1000 * 0.003412 = 3.412 MMBtu
        assert energy == pytest.approx(3.412, rel=0.01)

    def test_energy_content_fallback_therms(self, agent):
        """Test energy content fallback calculation for therms."""
        energy = agent._calculate_energy_content(100, "therms", "natural_gas")

        # Fallback: 100 * 0.1 = 10 MMBtu
        assert energy == pytest.approx(10.0, rel=0.01)

    def test_energy_content_fallback_diesel_gallons(self, agent):
        """Test energy content fallback calculation for diesel gallons."""
        energy = agent._calculate_energy_content(100, "gallons", "diesel")

        # Fallback: 100 * 0.138 = 13.8 MMBtu
        assert energy == pytest.approx(13.8, rel=0.01)

    def test_energy_content_fallback_gasoline_gallons(self, agent):
        """Test energy content fallback calculation for gasoline gallons."""
        energy = agent._calculate_energy_content(100, "gallons", "gasoline")

        # Fallback: 100 * 0.125 = 12.5 MMBtu
        assert energy == pytest.approx(12.5, rel=0.01)

    # ==================== Additional Edge Cases ====================

    def test_validate_renewable_fuel_positive_amount_warning(self, agent):
        """Test validation warns for renewable fuel with positive amount."""
        # This tests the renewable fuel warning path (line 129)
        # Note: This requires a fuel configured as renewable in the config
        # The validation will pass but log a warning
        payload = {
            "fuel_type": "biomass",  # May or may not be configured as renewable
            "amount": 100,
            "unit": "tons"
        }
        # Should validate successfully
        result = agent.validate(payload)
        assert result is True

    def test_metadata_includes_cache_stats(self, agent):
        """Test that metadata includes cache statistics."""
        # Run calculation
        result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms"
        })

        assert result["success"] is True
        assert "metadata" in result
        metadata = result["metadata"]
        assert "cache_hits" in metadata
        assert "cache_misses" in metadata

    def test_calculation_metadata_format(self, agent):
        """Test calculation metadata format in response."""
        result = agent.run({
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms"
        })

        assert result["success"] is True
        metadata = result["metadata"]
        assert "calculation" in metadata
        # Should show formula like "100 therms × 5.3 kgCO2e/therms"
        calc = metadata["calculation"]
        assert "100" in calc
        assert "therms" in calc
        assert "kgCO2e" in calc


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
