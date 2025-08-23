"""Tests for FuelAgent."""

import math
import pytest
from greenlang.agents.fuel_agent import FuelAgent
from tests.utils import assert_close


class TestFuelAgent:
    """Test suite for FuelAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = FuelAgent()
    
    def test_electricity_india_kwh(self, electricity_factors, agent_contract_validator):
        """Test electricity emissions calculation for India in kWh."""
        # Get expected factor from actual data
        expected_factor = electricity_factors.get("IN", {}).get("factor", 0.71)
        consumption = 1_500_000  # kWh
        expected_emissions = consumption * expected_factor
        
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": consumption, "unit": "kWh"},
            "country": "IN"
        })
        
        # Validate contract
        agent_contract_validator.validate_response(result, "FuelAgent")
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], expected_emissions, rel_tol=1e-9)
        assert result["data"]["emission_factor"] == expected_factor
        assert result["data"]["unit"] == "kgCO2e"
        
        # Validate provenance fields
        assert "source" in result["data"]
        assert "version" in result["data"] or "factor_version" in result["data"]
        assert "calculation_method" in result["data"] or "methodology" in result["data"]
    
    def test_electricity_us_kwh(self, electricity_factors, agent_contract_validator):
        """Test electricity emissions calculation for US in kWh."""
        expected_factor = electricity_factors.get("US", {}).get("factor", 0.385)
        consumption = 1_000_000
        expected_emissions = consumption * expected_factor
        
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": consumption, "unit": "kWh"},
            "country": "US"
        })
        
        agent_contract_validator.validate_response(result, "FuelAgent")
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], expected_emissions, rel_tol=1e-9)
        assert result["data"]["emission_factor"] == expected_factor
        assert result["data"]["unit"] == "kgCO2e"
    
    def test_electricity_eu_kwh(self, electricity_factors, agent_contract_validator):
        """Test electricity emissions calculation for EU in kWh."""
        expected_factor = electricity_factors.get("EU", {}).get("factor", 0.23)
        consumption = 1_000_000
        expected_emissions = consumption * expected_factor
        
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": consumption, "unit": "kWh"},
            "country": "EU"
        })
        
        agent_contract_validator.validate_response(result, "FuelAgent")
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], expected_emissions, rel_tol=1e-9)
        assert result["data"]["emission_factor"] == expected_factor
    
    def test_diesel_liters(self):
        """Test diesel emissions calculation in liters."""
        result = self.agent.run({
            "fuel_type": "diesel",
            "consumption": {"value": 1000, "unit": "liters"},
            "country": "IN"  # Country doesn't matter for diesel
        })
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], 2680.0, rel_tol=1e-9)
        assert result["data"]["emission_factor"] == 2.68
        assert result["data"]["unit"] == "kgCO2e"
    
    def test_natural_gas_therms(self):
        """Test natural gas emissions calculation in therms."""
        result = self.agent.run({
            "fuel_type": "natural_gas",
            "consumption": {"value": 10000, "unit": "therms"},
            "country": "US"
        })
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], 53000.0, rel_tol=1e-9)
        assert result["data"]["emission_factor"] == 5.3
    
    def test_natural_gas_m3(self):
        """Test natural gas emissions calculation in cubic meters."""
        result = self.agent.run({
            "fuel_type": "natural_gas",
            "consumption": {"value": 1000, "unit": "m3"},
            "country": "EU"
        })
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], 1880.0, rel_tol=1e-9)
        assert result["data"]["emission_factor"] == 1.88
    
    def test_zero_consumption(self):
        """Test zero consumption returns zero emissions."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 0, "unit": "kWh"},
            "country": "IN"
        })
        
        assert result["success"] is True
        assert result["data"]["co2e_emissions_kg"] == 0
        assert "note" in result["data"]
    
    def test_very_large_consumption(self):
        """Test very large consumption values."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 1e9, "unit": "kWh"},  # 1 billion kWh
            "country": "IN"
        })
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], 710_000_000.0, rel_tol=1e-9)
    
    def test_negative_consumption_rejected(self):
        """Test that negative consumption is rejected."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": -10, "unit": "kWh"},
            "country": "US"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["ValidationError", "ValueError"]
        assert "negative" in result["error"]["message"].lower()
    
    def test_unknown_fuel_type_rejected(self):
        """Test that unknown fuel types are rejected."""
        result = self.agent.run({
            "fuel_type": "unknown_fuel",
            "consumption": {"value": 100, "unit": "kWh"},
            "country": "US"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["ValidationError", "DataNotFoundError"]
        assert "fuel type" in result["error"]["message"].lower()
    
    def test_unknown_country_for_electricity(self):
        """Test that unknown countries are rejected for electricity."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 1000, "unit": "kWh"},
            "country": "XX"  # Invalid country code
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["DataNotFoundError", "ValidationError"]
        assert "country" in result["error"]["message"].lower()
    
    def test_invalid_unit_for_fuel_type(self):
        """Test that invalid units for fuel types are rejected."""
        result = self.agent.run({
            "fuel_type": "diesel",
            "consumption": {"value": 100, "unit": "kWh"},  # Wrong unit for diesel
            "country": "IN"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["ValidationError", "UnitError"]
        assert "unit" in result["error"]["message"].lower()
    
    def test_electricity_with_mwh_unit(self):
        """Test electricity with MWh unit conversion."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 1500, "unit": "MWh"},  # 1500 MWh = 1,500,000 kWh
            "country": "IN"
        })
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], 1_065_000.0, rel_tol=1e-9)
    
    def test_pv_offset_if_supported(self):
        """Test PV offset calculation if supported."""
        result = self.agent.run({
            "fuel_type": "solar_pv",
            "consumption": {"value": 100000, "unit": "kWh"},
            "country": "IN"
        })
        
        # If PV is supported, it should return negative or zero emissions
        if result["success"]:
            assert result["data"]["co2e_emissions_kg"] <= 0
            assert "offset" in result["data"] or "avoided" in result["data"]
    
    @pytest.mark.parametrize("fuel,unit,factor", [
        ("electricity", "kWh", 0.71),  # IN
        ("natural_gas", "therms", 5.3),
        ("diesel", "liters", 2.68),
        ("gasoline", "liters", 2.31),
    ])
    def test_fuel_factors_accuracy(self, fuel, unit, factor):
        """Test that emission factors match expected values."""
        result = self.agent.run({
            "fuel_type": fuel,
            "consumption": {"value": 1, "unit": unit},
            "country": "IN" if fuel == "electricity" else "global"
        })
        
        if result["success"]:
            assert_close(result["data"]["emission_factor"], factor, rel_tol=1e-6)
    
    def test_precision_maintained(self):
        """Test that internal calculations maintain precision."""
        # Test with a value that would show rounding errors
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 123456.789, "unit": "kWh"},
            "country": "IN"
        })
        
        assert result["success"] is True
        expected = 123456.789 * 0.71
        assert_close(result["data"]["co2e_emissions_kg"], expected, rel_tol=1e-12)
    
    def test_duplicate_fuel_aggregation(self):
        """Test handling of duplicate fuel entries if agent supports batching."""
        # This test assumes the agent can handle multiple entries
        result = self.agent.run({
            "fuels": [
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 1000, "unit": "kWh"},
                    "country": "IN"
                },
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 500, "unit": "kWh"},
                    "country": "IN"
                }
            ]
        })
        
        # If batching is supported
        if result["success"] and "total" in result["data"]:
            assert_close(result["data"]["total"]["co2e_emissions_kg"], 1065.0, rel_tol=1e-9)