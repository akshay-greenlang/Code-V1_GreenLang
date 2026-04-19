# -*- coding: utf-8 -*-
"""
Unit Tests for Fuel Agent

Tests FuelAgent - converts fuel consumption to emissions.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch


class TestFuelAgent:
    """Test FuelAgent functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test agent"""
        try:
            from greenlang.agents.fuel_agent import FuelAgent
            self.agent = FuelAgent()
        except ImportError:
            pytest.skip("FuelAgent not available")

    def test_process_diesel_consumption(self):
        """Test processing diesel fuel consumption"""
        input_data = {
            "fuel_type": "diesel",
            "fuel_quantity": 100,
            "fuel_unit": "liters",
            "region": "US"
        }

        result = self.agent.process(input_data)

        assert result["status"] == "success"
        assert "emissions_kg_co2e" in result
        assert result["emissions_kg_co2e"] > 0

    def test_process_natural_gas_consumption(self):
        """Test processing natural gas consumption"""
        input_data = {
            "fuel_type": "natural_gas",
            "fuel_quantity": 1000,
            "fuel_unit": "cubic_meters",
            "region": "US"
        }

        result = self.agent.process(input_data)

        assert result["status"] == "success"
        assert result["emissions_kg_co2e"] > 0

    @pytest.mark.parametrize("fuel,quantity,unit,expected_min", [
        ("diesel", 100, "liters", 250),
        ("gasoline", 100, "liters", 220),
        ("natural_gas", 1000, "cubic_meters", 1800),
        ("coal", 1000, "kg", 3000),
    ])
    def test_fuel_emission_calculations(self, fuel, quantity, unit, expected_min):
        """Test various fuel emission calculations"""
        input_data = {
            "fuel_type": fuel,
            "fuel_quantity": quantity,
            "fuel_unit": unit,
            "region": "US"
        }

        result = self.agent.process(input_data)

        assert result["emissions_kg_co2e"] >= expected_min

    def test_validation_missing_fuel_type(self):
        """Test validation fails when fuel_type missing"""
        input_data = {
            "fuel_quantity": 100,
            "fuel_unit": "liters"
        }

        with pytest.raises(ValueError, match="fuel_type"):
            self.agent.process(input_data)

    def test_validation_negative_quantity(self):
        """Test validation fails for negative quantity"""
        input_data = {
            "fuel_type": "diesel",
            "fuel_quantity": -100,
            "fuel_unit": "liters"
        }

        with pytest.raises(ValueError, match="negative"):
            self.agent.process(input_data)

    def test_provenance_tracking(self):
        """Test provenance hash is generated"""
        input_data = {
            "fuel_type": "diesel",
            "fuel_quantity": 100,
            "fuel_unit": "liters",
            "region": "US"
        }

        result = self.agent.process(input_data)

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_deterministic_calculation(self):
        """Test calculation is deterministic"""
        input_data = {
            "fuel_type": "diesel",
            "fuel_quantity": 100,
            "fuel_unit": "liters",
            "region": "US"
        }

        result1 = self.agent.process(input_data)
        result2 = self.agent.process(input_data)

        assert result1["emissions_kg_co2e"] == result2["emissions_kg_co2e"]
        assert result1["provenance_hash"] == result2["provenance_hash"]


class TestGridFactorAgent:
    """Test GridFactorAgent functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test agent"""
        try:
            from greenlang.agents.grid_factor_agent import GridFactorAgent
            self.agent = GridFactorAgent()
        except ImportError:
            pytest.skip("GridFactorAgent not available")

    def test_process_electricity_consumption(self):
        """Test processing electricity consumption"""
        input_data = {
            "electricity_kwh": 1000,
            "region": "US-CA",
            "calculation_method": "location_based"
        }

        result = self.agent.process(input_data)

        assert result["status"] == "success"
        assert "emissions_kg_co2e" in result

    def test_location_vs_market_based(self):
        """Test location-based vs market-based calculations"""
        input_data_location = {
            "electricity_kwh": 1000,
            "region": "US-CA",
            "calculation_method": "location_based"
        }

        input_data_market = {
            "electricity_kwh": 1000,
            "region": "US-CA",
            "calculation_method": "market_based",
            "green_energy_pct": 50
        }

        result_location = self.agent.process(input_data_location)
        result_market = self.agent.process(input_data_market)

        # Market-based should be lower with 50% green energy
        assert result_market["emissions_kg_co2e"] < result_location["emissions_kg_co2e"]


class TestCarbonAgentAI:
    """Test CarbonAgentAI functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test agent with mocked LLM"""
        try:
            from greenlang.agents.carbon_agent_ai import CarbonAgentAI
            self.agent = CarbonAgentAI()
        except ImportError:
            pytest.skip("CarbonAgentAI not available")

    @patch('greenlang.agents.carbon_agent_ai.LLMProvider')
    def test_ai_agent_with_mock_llm(self, mock_llm):
        """Test AI agent with mocked LLM provider"""
        # Mock LLM response
        mock_llm.return_value.generate.return_value = {
            "choices": [{
                "message": {
                    "content": '{"emissions_kg_co2e": 123.45, "confidence": 0.95}'
                }
            }]
        }

        input_data = {
            "description": "Office building electricity usage",
            "value": 1000,
            "unit": "kwh"
        }

        result = self.agent.process(input_data)

        assert result["status"] == "success"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
