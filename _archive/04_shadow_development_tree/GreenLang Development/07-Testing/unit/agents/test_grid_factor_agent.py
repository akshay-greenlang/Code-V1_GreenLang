# -*- coding: utf-8 -*-
"""Tests for GridFactorAgent."""

import pytest
from greenlang.agents.grid_factor_agent import GridFactorAgent


class TestGridFactorAgent:
    """Test suite for GridFactorAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = GridFactorAgent()
    
    @pytest.mark.parametrize("country,expected", [
        ("IN", 0.71),
        ("EU", 0.23),
        ("US", 0.385),
        ("CN", 0.555),
    ])
    def test_grid_factors(self, country, expected):
        """Test grid emission factors for different countries."""
        result = self.agent.run({
            "country": country,
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        assert result["success"] is True
        assert abs(result["data"]["emission_factor"] - expected) < 1e-12
        assert result["data"]["unit"] == "kgCO2e/kWh"
        assert result["data"]["country"] == country
        
        # Check provenance fields
        assert "source" in result["data"]
        assert "version" in result["data"]
        assert "last_updated" in result["data"]
    
    def test_unsupported_country_error(self):
        """Test error handling for unsupported countries."""
        result = self.agent.run({
            "country": "XX",  # Invalid country code
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["DataNotFoundError", "ValidationError"]
        assert "country" in result["error"]["message"].lower()
        # Error should list acceptable options
        assert any(x in result["error"]["message"] for x in ["IN", "US", "EU", "CN"])
    
    def test_unsupported_fuel_type(self):
        """Test error handling for unsupported fuel types."""
        result = self.agent.run({
            "country": "IN",
            "fuel_type": "unknown_fuel",
            "unit": "kWh"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["DataNotFoundError", "ValidationError"]
        assert "fuel" in result["error"]["message"].lower()
    
    def test_unsupported_unit(self):
        """Test error handling for unsupported units."""
        result = self.agent.run({
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "invalid_unit"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["DataNotFoundError", "ValidationError", "UnitError"]
        assert "unit" in result["error"]["message"].lower()
        # Error should list acceptable options
        assert "kWh" in result["error"]["message"]
    
    def test_electricity_mwh_unit(self):
        """Test electricity factor retrieval with MWh unit."""
        result = self.agent.run({
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "MWh"
        })
        
        assert result["success"] is True
        # Factor should be adjusted for MWh (710 instead of 0.71)
        assert abs(result["data"]["emission_factor"] - 710.0) < 1e-9
        assert result["data"]["unit"] == "kgCO2e/MWh"
    
    def test_natural_gas_global(self):
        """Test natural gas global emission factor."""
        result = self.agent.run({
            "country": "global",
            "fuel_type": "natural_gas",
            "unit": "therms"
        })
        
        assert result["success"] is True
        assert abs(result["data"]["emission_factor"] - 5.3) < 1e-12
        assert result["data"]["unit"] == "kgCO2e/therm"
    
    def test_diesel_global(self):
        """Test diesel global emission factor."""
        result = self.agent.run({
            "country": "global",
            "fuel_type": "diesel",
            "unit": "liters"
        })
        
        assert result["success"] is True
        assert abs(result["data"]["emission_factor"] - 2.68) < 1e-12
        assert result["data"]["unit"] == "kgCO2e/liter"
    
    def test_precision_preserved(self):
        """Test that full precision is preserved in factors."""
        result = self.agent.run({
            "country": "US",
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        assert result["success"] is True
        # Check that the factor maintains full precision
        factor = result["data"]["emission_factor"]
        assert factor == 0.385  # Exact value, not rounded
    
    def test_metadata_fields(self):
        """Test that all required metadata fields are present."""
        result = self.agent.run({
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Required fields
        assert "emission_factor" in data
        assert "unit" in data
        assert "country" in data
        assert "fuel_type" in data
        
        # Provenance fields
        assert "source" in data
        assert "version" in data
        assert "last_updated" in data
        
        # Optional but useful fields
        if "confidence" in data:
            assert data["confidence"] in ["high", "medium", "low"]
        if "methodology" in data:
            assert isinstance(data["methodology"], str)
    
    @pytest.mark.parametrize("country", ["IN", "US", "EU", "CN"])
    def test_consistent_structure(self, country):
        """Test that response structure is consistent across countries."""
        result = self.agent.run({
            "country": country,
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        assert result["success"] is True
        assert set(result["data"].keys()) >= {
            "emission_factor", "unit", "country", "fuel_type",
            "source", "version", "last_updated"
        }
    
    def test_case_insensitive_country(self):
        """Test that country codes are case-insensitive."""
        result_upper = self.agent.run({
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        result_lower = self.agent.run({
            "country": "in",
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        if result_lower["success"]:
            assert result_upper["data"]["emission_factor"] == result_lower["data"]["emission_factor"]
    
    def test_regional_variations(self):
        """Test support for regional variations within countries if available."""
        # Test US state-level factors if supported
        result = self.agent.run({
            "country": "US",
            "state": "CA",  # California
            "fuel_type": "electricity",
            "unit": "kWh"
        })
        
        # If regional data is supported, CA should have different factor than US average
        if result["success"] and "state" in result["data"]:
            assert result["data"]["state"] == "CA"
            # California typically has lower emissions than US average
            assert result["data"]["emission_factor"] < 0.385
    
    def test_historical_factors_if_supported(self):
        """Test retrieval of historical factors if supported."""
        result = self.agent.run({
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "kWh",
            "year": 2023
        })
        
        # If historical data is supported
        if result["success"] and "year" in result["data"]:
            assert result["data"]["year"] == 2023
            assert "emission_factor" in result["data"]
    
    def test_forecast_factors_if_supported(self):
        """Test retrieval of forecast factors if supported."""
        result = self.agent.run({
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "kWh",
            "year": 2030,
            "scenario": "net-zero"
        })
        
        # If forecast data is supported
        if result["success"] and "scenario" in result["data"]:
            assert result["data"]["scenario"] == "net-zero"
            assert result["data"]["year"] == 2030
            # Future net-zero scenario should have lower emissions
            assert result["data"]["emission_factor"] < 0.71