# -*- coding: utf-8 -*-
"""Comprehensive test suite for CarbonAgent.

Tests cover:
- Unit tests: Aggregation, calculations, breakdowns, categorization
- Integration tests: Realistic scenarios with multiple emission sources
- Determinism tests: Reproducibility and consistency
- Boundary tests: Edge cases, invalid inputs, extreme values

Test Data:
- Synthetic emission sources with various fuel types
- Realistic building and industrial facility scenarios
- Edge cases (zero, single, large-scale emissions)

Target: 85%+ code coverage

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from copy import deepcopy

from greenlang.agents.carbon_agent import CarbonAgent
from greenlang.agents.base import AgentResult, AgentConfig
from greenlang.determinism import deterministic_random


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def agent():
    """Create CarbonAgent instance."""
    return CarbonAgent()


@pytest.fixture
def agent_custom_config():
    """Create CarbonAgent with custom config."""
    config = AgentConfig(
        name="CustomCarbonAgent",
        description="Custom carbon aggregator for testing",
        version="1.0.0",
    )
    return CarbonAgent(config)


@pytest.fixture
def single_emission():
    """Single emission source."""
    return {
        "fuel_type": "Natural Gas",
        "co2e_emissions_kg": 1500.0,
        "quantity": 100,
        "unit": "therms",
    }


@pytest.fixture
def two_emissions():
    """Two emission sources."""
    return [
        {
            "fuel_type": "Electricity",
            "co2e_emissions_kg": 2000.0,
            "quantity": 5000,
            "unit": "kWh",
        },
        {
            "fuel_type": "Natural Gas",
            "co2e_emissions_kg": 1500.0,
            "quantity": 100,
            "unit": "therms",
        },
    ]


@pytest.fixture
def five_emissions():
    """Five emission sources."""
    return [
        {"fuel_type": "Electricity", "co2e_emissions_kg": 3000.0},
        {"fuel_type": "Natural Gas", "co2e_emissions_kg": 2000.0},
        {"fuel_type": "Diesel", "co2e_emissions_kg": 1500.0},
        {"fuel_type": "Gasoline", "co2e_emissions_kg": 1000.0},
        {"fuel_type": "Coal", "co2e_emissions_kg": 500.0},
    ]


@pytest.fixture
def ten_emissions():
    """Ten emission sources with various magnitudes."""
    return [
        {"fuel_type": "Electricity - Grid", "co2e_emissions_kg": 5000.0},
        {"fuel_type": "Natural Gas - Heating", "co2e_emissions_kg": 3500.0},
        {"fuel_type": "Natural Gas - Cooking", "co2e_emissions_kg": 500.0},
        {"fuel_type": "Diesel - Generator", "co2e_emissions_kg": 2000.0},
        {"fuel_type": "Gasoline - Vehicles", "co2e_emissions_kg": 1800.0},
        {"fuel_type": "LPG - Backup", "co2e_emissions_kg": 700.0},
        {"fuel_type": "Coal - Boiler", "co2e_emissions_kg": 4000.0},
        {"fuel_type": "Fuel Oil", "co2e_emissions_kg": 1200.0},
        {"fuel_type": "Propane", "co2e_emissions_kg": 900.0},
        {"fuel_type": "Biomass", "co2e_emissions_kg": 300.0},
    ]


@pytest.fixture
def building_emissions():
    """Realistic commercial building emissions."""
    return [
        {
            "fuel_type": "Electricity - Grid",
            "co2e_emissions_kg": 15000.0,
            "scope": 2,
            "category": "Purchased Electricity",
        },
        {
            "fuel_type": "Natural Gas - HVAC",
            "co2e_emissions_kg": 8000.0,
            "scope": 1,
            "category": "Stationary Combustion",
        },
        {
            "fuel_type": "Diesel - Emergency Generator",
            "co2e_emissions_kg": 500.0,
            "scope": 1,
            "category": "Stationary Combustion",
        },
    ]


@pytest.fixture
def industrial_emissions():
    """Industrial facility with multiple fuel types."""
    return [
        {"fuel_type": "Coal - Main Boiler", "co2e_emissions_kg": 50000.0, "scope": 1},
        {"fuel_type": "Natural Gas - Process Heat", "co2e_emissions_kg": 25000.0, "scope": 1},
        {"fuel_type": "Electricity - Motors", "co2e_emissions_kg": 30000.0, "scope": 2},
        {"fuel_type": "Diesel - Forklifts", "co2e_emissions_kg": 5000.0, "scope": 1},
        {"fuel_type": "LPG - Process", "co2e_emissions_kg": 3000.0, "scope": 1},
        {"fuel_type": "Fuel Oil - Backup", "co2e_emissions_kg": 2000.0, "scope": 1},
    ]


@pytest.fixture
def large_scale_emissions():
    """Large-scale aggregation (100+ sources)."""
    emissions = []
    np.random.seed(42)

    fuel_types = [
        "Electricity", "Natural Gas", "Diesel", "Coal",
        "Gasoline", "LPG", "Fuel Oil", "Propane"
    ]

    for i in range(100):
        emissions.append({
            "fuel_type": f"{fuel_types[i % len(fuel_types)]} - Unit {i}",
            "co2e_emissions_kg": np.random.uniform(100, 10000),
        })

    return emissions


@pytest.fixture
def emissions_with_scopes():
    """Emissions with Scope 1, 2, 3 categorization."""
    return [
        {"fuel_type": "Natural Gas", "co2e_emissions_kg": 5000.0, "scope": 1},
        {"fuel_type": "Diesel", "co2e_emissions_kg": 3000.0, "scope": 1},
        {"fuel_type": "Electricity", "co2e_emissions_kg": 10000.0, "scope": 2},
        {"fuel_type": "Employee Commuting", "co2e_emissions_kg": 2000.0, "scope": 3},
        {"fuel_type": "Business Travel", "co2e_emissions_kg": 1500.0, "scope": 3},
    ]


# ==============================================================================
# Test Class 1: Unit Tests - Basic Functionality (15+ tests)
# ==============================================================================

class TestCarbonAgentUnit:
    """Unit tests for CarbonAgent core functionality."""

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.config.name == "CarbonAgent"
        assert agent.config.description == "Aggregates emissions and provides carbon footprint"
        assert agent.config.enabled is True

    def test_agent_custom_config(self, agent_custom_config):
        """Test agent with custom configuration."""
        assert agent_custom_config.config.name == "CustomCarbonAgent"
        assert agent_custom_config.config.version == "1.0.0"

    def test_validate_input_valid(self, agent, two_emissions):
        """Test validation passes with valid input."""
        input_data = {"emissions": two_emissions}
        assert agent.validate_input(input_data) is True

    def test_validate_input_missing_emissions(self, agent):
        """Test validation fails without emissions key."""
        input_data = {"other_field": "value"}
        assert agent.validate_input(input_data) is False

    def test_validate_input_not_list(self, agent):
        """Test validation fails when emissions is not a list."""
        input_data = {"emissions": "not a list"}
        assert agent.validate_input(input_data) is False

    def test_validate_input_empty_dict(self, agent):
        """Test validation fails with empty dict."""
        input_data = {}
        assert agent.validate_input(input_data) is False

    def test_empty_emissions_list(self, agent):
        """Test handling of empty emissions list."""
        input_data = {"emissions": []}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 0
        assert result.data["total_co2e_tons"] == 0
        assert result.data["emissions_breakdown"] == []
        assert result.data["summary"] == "No emissions data provided"

    def test_single_emission_aggregation(self, agent, single_emission):
        """Test aggregation of single emission source."""
        input_data = {"emissions": [single_emission]}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 1500.0
        assert result.data["total_co2e_tons"] == 1.5
        assert len(result.data["emissions_breakdown"]) == 1

    def test_two_emissions_aggregation(self, agent, two_emissions):
        """Test aggregation of two emission sources."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 3500.0
        assert result.data["total_co2e_tons"] == 3.5
        assert len(result.data["emissions_breakdown"]) == 2

    def test_five_emissions_aggregation(self, agent, five_emissions):
        """Test aggregation of five emission sources."""
        input_data = {"emissions": five_emissions}
        result = agent.execute(input_data)

        total = 3000.0 + 2000.0 + 1500.0 + 1000.0 + 500.0
        assert result.success is True
        assert result.data["total_co2e_kg"] == total
        assert result.data["total_co2e_tons"] == total / 1000
        assert len(result.data["emissions_breakdown"]) == 5

    def test_ten_emissions_aggregation(self, agent, ten_emissions):
        """Test aggregation of ten emission sources."""
        input_data = {"emissions": ten_emissions}
        result = agent.execute(input_data)

        expected_total = sum(e["co2e_emissions_kg"] for e in ten_emissions)
        assert result.success is True
        assert result.data["total_co2e_kg"] == expected_total
        assert len(result.data["emissions_breakdown"]) == 10

    def test_breakdown_structure(self, agent, two_emissions):
        """Test emissions breakdown structure."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        breakdown = result.data["emissions_breakdown"]
        assert len(breakdown) == 2

        for item in breakdown:
            assert "source" in item
            assert "co2e_kg" in item
            assert "co2e_tons" in item
            assert "percentage" in item

    def test_percentage_calculations(self, agent, two_emissions):
        """Test percentage calculations are correct."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        breakdown = result.data["emissions_breakdown"]

        # Check percentages sum to 100
        total_percentage = sum(item["percentage"] for item in breakdown)
        assert abs(total_percentage - 100.0) < 0.01

        # Check individual percentages
        assert breakdown[0]["percentage"] == round((2000.0 / 3500.0) * 100, 2)
        assert breakdown[1]["percentage"] == round((1500.0 / 3500.0) * 100, 2)

    def test_unit_conversion_kg_to_tons(self, agent, two_emissions):
        """Test kg to tons conversion."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        for item in result.data["emissions_breakdown"]:
            expected_tons = item["co2e_kg"] / 1000
            assert item["co2e_tons"] == expected_tons

    def test_summary_generation(self, agent, two_emissions):
        """Test summary string generation."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        summary = result.data["summary"]
        assert "Total carbon footprint" in summary
        assert "3.500 metric tons CO2e" in summary
        assert "Breakdown by source" in summary
        assert "Electricity" in summary
        assert "Natural Gas" in summary


# ==============================================================================
# Test Class 2: Integration Tests - Realistic Scenarios (8+ tests)
# ==============================================================================

class TestCarbonAgentIntegration:
    """Integration tests with realistic emission scenarios."""

    def test_building_scenario(self, agent, building_emissions):
        """Test realistic commercial building scenario."""
        input_data = {
            "emissions": building_emissions,
            "building_area": 10000,  # sqft
            "occupancy": 50,
        }
        result = agent.execute(input_data)

        assert result.success is True
        total_kg = 15000.0 + 8000.0 + 500.0
        assert result.data["total_co2e_kg"] == total_kg

        # Check carbon intensity
        intensity = result.data["carbon_intensity"]
        assert "per_sqft" in intensity
        assert "per_person" in intensity
        assert intensity["per_sqft"] == total_kg / 10000
        assert intensity["per_person"] == total_kg / 50

    def test_industrial_facility(self, agent, industrial_emissions):
        """Test industrial facility with multiple fuel types."""
        input_data = {"emissions": industrial_emissions}
        result = agent.execute(input_data)

        expected_total = 50000.0 + 25000.0 + 30000.0 + 5000.0 + 3000.0 + 2000.0
        assert result.success is True
        assert result.data["total_co2e_kg"] == expected_total
        assert result.data["total_co2e_tons"] == expected_total / 1000

        # Check breakdown sorted by emissions
        breakdown = result.data["emissions_breakdown"]
        emissions_values = [item["co2e_kg"] for item in breakdown]

        # Verify largest source is Coal
        assert breakdown[0]["source"] == "Coal - Main Boiler"
        assert breakdown[0]["co2e_kg"] == 50000.0

    def test_large_scale_aggregation(self, agent, large_scale_emissions):
        """Test aggregation of 100+ emission sources."""
        input_data = {"emissions": large_scale_emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert len(result.data["emissions_breakdown"]) == 100

        # Verify total matches sum of individual sources
        expected_total = sum(e["co2e_emissions_kg"] for e in large_scale_emissions)
        assert abs(result.data["total_co2e_kg"] - expected_total) < 0.01

        # Verify metadata
        assert result.metadata["num_sources"] == 100

    def test_scope_categorization(self, agent, emissions_with_scopes):
        """Test emissions with Scope 1, 2, 3 categorization."""
        input_data = {"emissions": emissions_with_scopes}
        result = agent.execute(input_data)

        assert result.success is True

        # Calculate expected totals by scope
        scope1_total = 5000.0 + 3000.0  # Natural Gas + Diesel
        scope2_total = 10000.0  # Electricity
        scope3_total = 2000.0 + 1500.0  # Commuting + Travel

        total = scope1_total + scope2_total + scope3_total
        assert result.data["total_co2e_kg"] == total

    def test_real_emission_data_structure(self, agent):
        """Test with realistic emission data structure."""
        emissions = [
            {
                "fuel_type": "Electricity",
                "co2e_emissions_kg": 12500.0,
                "quantity": 25000,
                "unit": "kWh",
                "emission_factor": 0.5,
                "region": "US",
                "scope": 2,
            },
            {
                "fuel_type": "Natural Gas",
                "co2e_emissions_kg": 6800.0,
                "quantity": 1200,
                "unit": "therms",
                "emission_factor": 5.667,
                "region": "US",
                "scope": 1,
            },
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 19300.0

    def test_mixed_emission_sources(self, agent):
        """Test mixed emission sources with various properties."""
        emissions = [
            {"fuel_type": "Grid Electricity", "co2e_emissions_kg": 8000.0},
            {"fuel_type": "Solar (offset)", "co2e_emissions_kg": -500.0},
            {"fuel_type": "Natural Gas", "co2e_emissions_kg": 5000.0},
            {"fuel_type": "Diesel", "co2e_emissions_kg": 2000.0},
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        # Note: negative emissions are handled as-is
        expected_total = 8000.0 + (-500.0) + 5000.0 + 2000.0
        assert result.data["total_co2e_kg"] == expected_total

    def test_carbon_intensity_with_area(self, agent, two_emissions):
        """Test carbon intensity per square foot."""
        input_data = {
            "emissions": two_emissions,
            "building_area": 5000,
        }
        result = agent.execute(input_data)

        intensity = result.data["carbon_intensity"]
        assert "per_sqft" in intensity
        assert intensity["per_sqft"] == 3500.0 / 5000

    def test_carbon_intensity_with_occupancy(self, agent, two_emissions):
        """Test carbon intensity per person."""
        input_data = {
            "emissions": two_emissions,
            "occupancy": 25,
        }
        result = agent.execute(input_data)

        intensity = result.data["carbon_intensity"]
        assert "per_person" in intensity
        assert intensity["per_person"] == 3500.0 / 25


# ==============================================================================
# Test Class 3: Determinism Tests - Reproducibility (5+ tests)
# ==============================================================================

class TestCarbonAgentDeterminism:
    """Tests for deterministic behavior and reproducibility."""

    def test_same_input_same_output(self, agent, five_emissions):
        """Test same input produces same output."""
        input_data = {"emissions": five_emissions}

        result1 = agent.execute(input_data)
        result2 = agent.execute(input_data)

        assert result1.data["total_co2e_kg"] == result2.data["total_co2e_kg"]
        assert result1.data["total_co2e_tons"] == result2.data["total_co2e_tons"]
        assert result1.data["summary"] == result2.data["summary"]

        # Compare breakdowns
        for i in range(len(result1.data["emissions_breakdown"])):
            item1 = result1.data["emissions_breakdown"][i]
            item2 = result2.data["emissions_breakdown"][i]
            assert item1 == item2

    def test_floating_point_consistency(self, agent):
        """Test floating-point calculations are consistent."""
        emissions = [
            {"fuel_type": "Source A", "co2e_emissions_kg": 1234.567},
            {"fuel_type": "Source B", "co2e_emissions_kg": 9876.543},
            {"fuel_type": "Source C", "co2e_emissions_kg": 5555.555},
        ]

        input_data = {"emissions": emissions}

        # Run multiple times
        results = [agent.execute(input_data) for _ in range(5)]

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].data["total_co2e_kg"] == results[0].data["total_co2e_kg"]
            assert results[i].data["total_co2e_tons"] == results[0].data["total_co2e_tons"]

    def test_order_independence(self, agent, five_emissions):
        """Test aggregation is order-independent."""
        import random

        input_data1 = {"emissions": five_emissions}
        result1 = agent.execute(input_data1)

        # Shuffle emissions
        shuffled = five_emissions.copy()
        random.seed(42)
        deterministic_random().shuffle(shuffled)

        input_data2 = {"emissions": shuffled}
        result2 = agent.execute(input_data2)

        # Totals should be the same regardless of order
        assert result1.data["total_co2e_kg"] == result2.data["total_co2e_kg"]
        assert result1.data["total_co2e_tons"] == result2.data["total_co2e_tons"]

    def test_deep_copy_independence(self, agent, two_emissions):
        """Test deep copying doesn't affect results."""
        input_data1 = {"emissions": two_emissions}
        input_data2 = {"emissions": deepcopy(two_emissions)}

        result1 = agent.execute(input_data1)
        result2 = agent.execute(input_data2)

        assert result1.data["total_co2e_kg"] == result2.data["total_co2e_kg"]
        assert result1.data == result2.data

    def test_rounding_consistency(self, agent):
        """Test rounding is consistent across runs."""
        emissions = [
            {"fuel_type": "A", "co2e_emissions_kg": 1234.56789},
            {"fuel_type": "B", "co2e_emissions_kg": 9876.54321},
        ]

        input_data = {"emissions": emissions}

        # Run multiple times
        results = [agent.execute(input_data) for _ in range(10)]

        # All should have same rounded values
        for result in results:
            assert result.data["total_co2e_kg"] == 11111.11
            assert result.data["total_co2e_tons"] == 11.111


# ==============================================================================
# Test Class 4: Boundary Tests - Edge Cases (7+ tests)
# ==============================================================================

class TestCarbonAgentBoundary:
    """Boundary and edge case tests."""

    def test_zero_emissions(self, agent):
        """Test handling of zero emissions."""
        emissions = [
            {"fuel_type": "Renewable Energy", "co2e_emissions_kg": 0.0},
            {"fuel_type": "Solar Power", "co2e_emissions_kg": 0.0},
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 0.0
        assert result.data["total_co2e_tons"] == 0.0

        # Percentages should be 0 when total is 0
        for item in result.data["emissions_breakdown"]:
            assert item["percentage"] == 0

    def test_single_source(self, agent, single_emission):
        """Test single emission source."""
        input_data = {"emissions": [single_emission]}
        result = agent.execute(input_data)

        assert result.success is True
        assert len(result.data["emissions_breakdown"]) == 1

        # Single source should be 100%
        assert result.data["emissions_breakdown"][0]["percentage"] == 100.0

    def test_very_large_values(self, agent):
        """Test very large emission values."""
        emissions = [
            {"fuel_type": "Mega Plant", "co2e_emissions_kg": 1_000_000_000.0},
            {"fuel_type": "Large Facility", "co2e_emissions_kg": 500_000_000.0},
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 1_500_000_000.0
        assert result.data["total_co2e_tons"] == 1_500_000.0

    def test_very_small_values(self, agent):
        """Test very small emission values."""
        emissions = [
            {"fuel_type": "Micro Source", "co2e_emissions_kg": 0.001},
            {"fuel_type": "Tiny Source", "co2e_emissions_kg": 0.0001},
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        assert result.success is True
        # Result is rounded to 2 decimals by the agent
        assert result.data["total_co2e_kg"] == 0.0

    def test_missing_co2e_field(self, agent):
        """Test handling of missing co2e_emissions_kg field."""
        emissions = [
            {"fuel_type": "Electricity"},  # Missing co2e_emissions_kg
            {"fuel_type": "Natural Gas", "co2e_emissions_kg": 1000.0},
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        # Should treat missing as 0
        assert result.success is True
        assert result.data["total_co2e_kg"] == 1000.0

    def test_missing_fuel_type(self, agent):
        """Test handling of missing fuel_type field."""
        emissions = [
            {"co2e_emissions_kg": 1500.0},  # Missing fuel_type
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["emissions_breakdown"][0]["source"] == "Unknown"

    def test_non_dict_emission_items(self, agent):
        """Test handling of non-dict items in emissions list."""
        emissions = [
            {"fuel_type": "Electricity", "co2e_emissions_kg": 1000.0},
            "invalid_item",  # Not a dict
            {"fuel_type": "Natural Gas", "co2e_emissions_kg": 500.0},
        ]

        input_data = {"emissions": emissions}
        result = agent.execute(input_data)

        # Should skip non-dict items
        assert result.success is True
        assert result.data["total_co2e_kg"] == 1500.0
        assert len(result.data["emissions_breakdown"]) == 2

    def test_zero_building_area(self, agent, two_emissions):
        """Test carbon intensity with zero building area."""
        input_data = {
            "emissions": two_emissions,
            "building_area": 0,
        }
        result = agent.execute(input_data)

        intensity = result.data["carbon_intensity"]
        assert intensity["per_sqft"] == 0  # Should handle division by zero

    def test_zero_occupancy(self, agent, two_emissions):
        """Test carbon intensity with zero occupancy."""
        input_data = {
            "emissions": two_emissions,
            "occupancy": 0,
        }
        result = agent.execute(input_data)

        intensity = result.data["carbon_intensity"]
        assert intensity["per_person"] == 0  # Should handle division by zero

    def test_negative_building_area(self, agent, two_emissions):
        """Test carbon intensity with negative building area."""
        input_data = {
            "emissions": two_emissions,
            "building_area": -1000,
        }
        result = agent.execute(input_data)

        # Should not calculate intensity for invalid area
        intensity = result.data["carbon_intensity"]
        assert intensity["per_sqft"] == 0


# ==============================================================================
# Additional Tests - Summary and Metadata
# ==============================================================================

class TestCarbonAgentSummaryAndMetadata:
    """Tests for summary generation and metadata."""

    def test_summary_with_breakdown(self, agent, five_emissions):
        """Test summary includes breakdown."""
        input_data = {"emissions": five_emissions}
        result = agent.execute(input_data)

        summary = result.data["summary"]

        # Check for all sources in summary (sorted by emissions)
        assert "Electricity" in summary
        assert "Natural Gas" in summary
        assert "Diesel" in summary
        assert "Gasoline" in summary
        assert "Coal" in summary

    def test_summary_sorted_by_emissions(self, agent, five_emissions):
        """Test summary sources are sorted by emissions."""
        input_data = {"emissions": five_emissions}
        result = agent.execute(input_data)

        summary = result.data["summary"]

        # Highest emission source should appear first
        electricity_pos = summary.find("Electricity")
        coal_pos = summary.find("Coal")

        # Electricity (3000 kg) should appear before Coal (500 kg)
        assert electricity_pos < coal_pos

    def test_metadata_structure(self, agent, two_emissions):
        """Test metadata includes agent info."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        assert "metadata" in result.__dict__
        assert result.metadata["agent"] == "CarbonAgent"
        assert result.metadata["num_sources"] == 2

    def test_metadata_num_sources(self, agent, ten_emissions):
        """Test metadata tracks correct number of sources."""
        input_data = {"emissions": ten_emissions}
        result = agent.execute(input_data)

        assert result.metadata["num_sources"] == 10

    def test_result_success_flag(self, agent, two_emissions):
        """Test result success flag is set correctly."""
        input_data = {"emissions": two_emissions}
        result = agent.execute(input_data)

        assert result.success is True
        assert isinstance(result, AgentResult)


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
