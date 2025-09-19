"""Tests for BoilerAgent."""

import math
import sys
from pathlib import Path
import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.agents.boiler_agent import BoilerAgent
from test_utils import assert_close


class TestBoilerAgent:
    """Test suite for BoilerAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = BoilerAgent()
    
    def test_thermal_to_fuel_natural_gas(self, agent_contract_validator):
        """Test calculation from thermal output to fuel consumption for natural gas boiler."""
        result = self.agent.run({
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"},
            "efficiency": 0.95,
            "country": "US",
            "age": "new"
        })
        
        # Validate contract
        agent_contract_validator.validate_response(result, "BoilerAgent")
        
        assert result["success"] is True
        assert result["data"]["boiler_type"] == "condensing"
        assert result["data"]["fuel_type"] == "natural_gas"
        assert result["data"]["efficiency"] == 0.95
        assert result["data"]["thermal_efficiency_percent"] == 95.0
        
        # Check fuel calculation: 100 MMBtu thermal / 0.95 efficiency = 105.26 MMBtu fuel
        # 105.26 MMBtu * 10 = 1052.6 therms
        expected_fuel = (100 / 0.95) * 10  # Convert to therms
        assert_close(result["data"]["fuel_consumption_value"], expected_fuel, rel_tol=0.01)
        assert result["data"]["fuel_consumption_unit"] == "therms"
        
        # Check performance rating
        assert result["data"]["performance_rating"] == "Excellent"
    
    def test_fuel_to_thermal_oil_boiler(self, agent_contract_validator):
        """Test calculation from fuel consumption to thermal output for oil boiler."""
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "oil",
            "fuel_consumption": {"value": 1000, "unit": "gallons"},
            "efficiency": 0.80,
            "country": "US",
            "age": "medium"
        })
        
        agent_contract_validator.validate_response(result, "BoilerAgent")
        
        assert result["success"] is True
        assert result["data"]["fuel_consumption_value"] == 1000
        assert result["data"]["fuel_consumption_unit"] == "gallons"
        
        # Check thermal output: 1000 gallons / 7.15 gallons per MMBtu = 139.86 MMBtu fuel
        # 139.86 MMBtu * 0.80 efficiency = 111.89 MMBtu thermal
        expected_thermal = (1000 / 7.15) * 0.80
        assert_close(result["data"]["thermal_output_value"], expected_thermal, rel_tol=0.01)
        assert result["data"]["thermal_output_unit"] == "MMBtu"
        
        # Check performance rating (0.80 efficiency for oil = Good)
        assert result["data"]["performance_rating"] == "Good"
    
    def test_electric_heat_pump(self, agent_contract_validator):
        """Test electric heat pump with COP > 1."""
        result = self.agent.run({
            "boiler_type": "heat_pump",
            "fuel_type": "electricity",
            "thermal_output": {"value": 300000, "unit": "kWh"},
            "efficiency": 3.0,  # COP of 3.0
            "country": "US"
        })
        
        agent_contract_validator.validate_response(result, "BoilerAgent")
        
        assert result["success"] is True
        assert result["data"]["efficiency"] == 3.0
        
        # Heat pump with COP 3.0: 300000 kWh thermal / 3.0 = 100000 kWh electric
        expected_fuel = 300000 / 3.0
        assert_close(result["data"]["fuel_consumption_value"], expected_fuel, rel_tol=0.01)
        assert result["data"]["fuel_consumption_unit"] == "kWh"
        
        # Check performance rating (COP 3.0 = Excellent)
        assert result["data"]["performance_rating"] == "Excellent"
    
    def test_efficiency_defaults(self):
        """Test that default efficiencies are applied correctly."""
        # Test without providing efficiency
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "fuel_consumption": {"value": 500, "unit": "therms"},
            "age": "old",
            "country": "US"
        })
        
        assert result["success"] is True
        # Should use default efficiency for standard natural gas old boiler
        assert result["data"]["efficiency"] == 0.75
    
    def test_biomass_boiler(self, agent_contract_validator):
        """Test biomass boiler emissions calculation."""
        result = self.agent.run({
            "boiler_type": "modern",
            "fuel_type": "biomass",
            "thermal_output": {"value": 50, "unit": "MMBtu"},
            "efficiency": 0.80,
            "country": "EU",
            "age": "new"
        })
        
        agent_contract_validator.validate_response(result, "BoilerAgent")
        
        assert result["success"] is True
        assert result["data"]["fuel_type"] == "biomass"
        assert result["data"]["efficiency"] == 0.80
        
        # Check fuel calculation
        expected_fuel = 50 / 0.80  # 62.5 MMBtu
        assert_close(result["data"]["fuel_consumption_value"], expected_fuel, rel_tol=0.01)
    
    def test_recommendations_poor_efficiency(self):
        """Test that appropriate recommendations are generated for poor efficiency."""
        result = self.agent.run({
            "boiler_type": "low_efficiency",
            "fuel_type": "oil",
            "fuel_consumption": {"value": 2000, "unit": "gallons"},
            "efficiency": 0.60,
            "age": "old",
            "country": "US"
        })
        
        assert result["success"] is True
        recommendations = result["data"]["recommendations"]
        
        # Should recommend replacement for low efficiency
        assert len(recommendations) > 0
        high_priority_recs = [r for r in recommendations if r["priority"] == "high"]
        assert len(high_priority_recs) > 0
        
        # Check for replacement recommendation
        replacement_rec = next((r for r in recommendations 
                               if "Replace boiler" in r["action"]), None)
        assert replacement_rec is not None
    
    def test_fuel_switching_recommendation(self):
        """Test fuel switching recommendations for oil boilers."""
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "oil",
            "fuel_consumption": {"value": 1000, "unit": "gallons"},
            "country": "US"
        })
        
        assert result["success"] is True
        recommendations = result["data"]["recommendations"]
        
        # Should recommend switching from oil to natural gas
        gas_switch_rec = next((r for r in recommendations 
                              if "natural gas" in r["action"].lower()), None)
        assert gas_switch_rec is not None
    
    def test_intensity_metrics(self):
        """Test fuel and emission intensity calculations."""
        result = self.agent.run({
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"},
            "efficiency": 0.90,
            "country": "US"
        })
        
        assert result["success"] is True
        
        # Fuel intensity = fuel consumption / thermal output
        fuel_consumption = (100 / 0.90) * 10  # therms
        expected_fuel_intensity = fuel_consumption / 100
        assert_close(result["data"]["fuel_intensity"], expected_fuel_intensity, rel_tol=0.01)
        
        # Emission intensity = emissions / thermal output
        assert result["data"]["emission_intensity"] > 0
    
    def test_validation_errors(self):
        """Test input validation."""
        # Missing boiler type
        result = self.agent.run({
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"}
        })
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        
        # Missing both thermal output and fuel consumption
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "natural_gas"
        })
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        
        # Invalid efficiency
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"},
            "efficiency": 150  # Invalid - over 100%
        })
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
    
    def test_percentage_efficiency_conversion(self):
        """Test that efficiency given as percentage is converted correctly."""
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"},
            "efficiency": 85  # Given as percentage
        })
        
        assert result["success"] is True
        # Should be converted to 0.85
        assert result["data"]["efficiency"] == 0.85
        assert result["data"]["thermal_efficiency_percent"] == 85.0
    
    def test_coal_boiler_high_emissions(self):
        """Test coal boiler with expected high emissions."""
        result = self.agent.run({
            "boiler_type": "stoker",
            "fuel_type": "coal",
            "fuel_consumption": {"value": 10, "unit": "tons"},
            "efficiency": 0.70,
            "age": "old",
            "country": "CN"
        })
        
        assert result["success"] is True
        assert result["data"]["fuel_type"] == "coal"
        
        # Should have recommendations for switching from coal
        recommendations = result["data"]["recommendations"]
        coal_switch_rec = next((r for r in recommendations 
                               if "Switch to cleaner fuel" in r["action"]), None)
        assert coal_switch_rec is not None
        assert coal_switch_rec["priority"] == "high"
    
    def test_thermal_unit_conversions(self):
        """Test various thermal unit conversions."""
        # Test kWh to MMBtu conversion
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 293071, "unit": "kWh"},  # ~1 MMBtu
            "efficiency": 0.80
        })
        
        assert result["success"] is True
        # 293071 kWh * 0.003412 = ~1 MMBtu
        expected_fuel = (293071 * 0.003412 / 0.80) * 10  # therms
        assert_close(result["data"]["fuel_consumption_value"], expected_fuel, rel_tol=0.02)
        
        # Test MJ to MMBtu conversion
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 1055056, "unit": "MJ"},  # ~1 MMBtu
            "efficiency": 0.80
        })
        
        assert result["success"] is True
        # 1055056 MJ * 0.000948 = ~1 MMBtu
        expected_fuel = (1055056 * 0.000948 / 0.80) * 10  # therms
        assert_close(result["data"]["fuel_consumption_value"], expected_fuel, rel_tol=0.02)
    
    def test_district_heating_equivalent(self):
        """Test district heating system (treated as high-efficiency boiler)."""
        result = self.agent.run({
            "boiler_type": "condensing",
            "fuel_type": "district_heating",
            "thermal_output": {"value": 5000, "unit": "kWh"},
            "efficiency": 0.95,  # District heating typically very efficient
            "country": "EU"
        })
        
        assert result["success"] is True
        assert result["data"]["efficiency"] == 0.95
        assert result["data"]["performance_rating"] in ["Excellent", "Good"]
    
    def test_metadata_fields(self):
        """Test that metadata fields are properly included."""
        result = self.agent.run({
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"},
            "country": "US"
        })
        
        assert result["success"] is True
        assert "source" in result["data"]
        assert "version" in result["data"]
        assert "last_updated" in result["data"]
        assert "metadata" in result
        assert result["metadata"]["agent_id"] == "boiler"
    
    def test_propane_boiler(self):
        """Test propane boiler calculations."""
        result = self.agent.run({
            "boiler_type": "standard",
            "fuel_type": "propane",
            "fuel_consumption": {"value": 500, "unit": "gallons"},
            "efficiency": 0.85,
            "country": "US"
        })
        
        assert result["success"] is True
        assert result["data"]["fuel_type"] == "propane"
        
        # Check thermal output calculation
        # 500 gallons / 10.92 gallons per MMBtu = 45.79 MMBtu fuel
        # 45.79 MMBtu * 0.85 efficiency = 38.92 MMBtu thermal
        expected_thermal = (500 / 10.92) * 0.85
        assert_close(result["data"]["thermal_output_value"], expected_thermal, rel_tol=0.02)