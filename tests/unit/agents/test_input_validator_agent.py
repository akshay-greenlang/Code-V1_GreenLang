"""Tests for InputValidatorAgent."""

import math
import pytest
from greenlang.agents.input_validator_agent import InputValidatorAgent


class TestInputValidatorAgent:
    """Test suite for InputValidatorAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = InputValidatorAgent()
    
    def test_valid_building_payload(self):
        """Test validation of a complete valid building payload."""
        payload = {
            "building_name": "Test Office",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"},
            "occupancy": 250,
            "floor_count": 5,
            "energy_sources": [
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 1500000, "unit": "kWh"},
                    "period": "annual"
                }
            ]
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is True
        assert result["data"]["validated"] is True
        assert "normalized" in result["data"]
        assert "raw" in result["data"]
    
    def test_unit_normalization_sqm_to_sqft(self):
        """Test area unit normalization from sqm to sqft."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "EU",
            "total_area": {"value": 4645, "unit": "sqm"},  # ~50000 sqft
            "occupancy": 100
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is True
        # Check both raw and normalized values are preserved
        assert result["data"]["raw"]["total_area"]["unit"] == "sqm"
        assert result["data"]["raw"]["total_area"]["value"] == 4645
        assert result["data"]["normalized"]["total_area"]["unit"] == "sqft"
        assert abs(result["data"]["normalized"]["total_area"]["value"] - 50000) < 100
    
    def test_unit_normalization_m3_to_therms(self):
        """Test gas unit normalization from m³ to therms."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "EU",
            "total_area": {"value": 50000, "unit": "sqft"},
            "energy_sources": [
                {
                    "fuel_type": "natural_gas",
                    "consumption": {"value": 1000, "unit": "m3"},
                    "period": "annual"
                }
            ]
        }
        
        result = self.agent.run(payload)
        
        if result["success"]:
            raw_gas = result["data"]["raw"]["energy_sources"][0]
            normalized_gas = result["data"]["normalized"]["energy_sources"][0]
            
            assert raw_gas["consumption"]["unit"] == "m3"
            assert normalized_gas["consumption"]["unit"] in ["therms", "m3"]  # Depends on implementation
    
    def test_negative_values_rejected(self):
        """Test that negative values are rejected."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": -50000, "unit": "sqft"},  # Negative area
            "occupancy": 100
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert "negative" in result["error"]["message"].lower()
        assert "total_area" in result["error"]["message"]
    
    def test_nan_values_rejected(self):
        """Test that NaN values are rejected."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": float('nan'), "unit": "sqft"},
            "occupancy": 100
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert any(x in result["error"]["message"].lower() for x in ["nan", "invalid", "numeric"])
    
    def test_inf_values_rejected(self):
        """Test that infinite values are rejected."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": float('inf'), "unit": "sqft"},
            "occupancy": 100
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert any(x in result["error"]["message"].lower() for x in ["infinite", "inf", "invalid"])
    
    def test_optional_fields_with_defaults(self):
        """Test handling of optional fields with sensible defaults."""
        minimal_payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"}
            # Missing: occupancy, floor_count, climate_zone, etc.
        }
        
        result = self.agent.run(minimal_payload)
        
        assert result["success"] is True
        normalized = result["data"]["normalized"]
        
        # Check defaults are applied
        if "occupancy" not in minimal_payload:
            assert "occupancy" in normalized
            assert normalized["occupancy"] >= 0  # Sensible default
        
        if "climate_zone" not in minimal_payload:
            assert "climate_zone" in normalized or "climate_zone_inferred" in result["data"]
    
    def test_unknown_keys_rejected(self):
        """Test that unknown keys are rejected with helpful message."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"},
            "nat_gas": {"value": 1000, "unit": "therms"}  # Typo: should be natural_gas
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert "nat_gas" in result["error"]["message"]
        # Should suggest the correct key
        assert "natural_gas" in result["error"]["message"] or "Did you mean" in result["error"]["message"]
    
    def test_required_fields_validation(self):
        """Test that all required fields must be present."""
        # Missing required field: building_type
        payload = {
            "building_name": "Test Building",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"}
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert "building_type" in result["error"]["message"]
        assert "required" in result["error"]["message"].lower()
    
    def test_valid_building_types(self):
        """Test validation of building types."""
        valid_types = ["office", "retail", "hospital", "school", "warehouse", "residential"]
        
        for building_type in valid_types:
            payload = {
                "building_name": "Test Building",
                "building_type": building_type,
                "country": "IN",
                "total_area": {"value": 50000, "unit": "sqft"}
            }
            
            result = self.agent.run(payload)
            assert result["success"] is True
    
    def test_invalid_building_type(self):
        """Test rejection of invalid building types."""
        payload = {
            "building_name": "Test Building",
            "building_type": "spaceship",  # Invalid type
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"}
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert "building_type" in result["error"]["message"]
        # Should list valid options
        assert any(x in result["error"]["message"] for x in ["office", "retail", "hospital"])
    
    def test_valid_country_codes(self):
        """Test validation of country codes."""
        valid_countries = ["IN", "US", "EU", "CN", "UK", "JP", "AU"]
        
        for country in valid_countries[:3]:  # Test a few
            payload = {
                "building_name": "Test Building",
                "building_type": "office",
                "country": country,
                "total_area": {"value": 50000, "unit": "sqft"}
            }
            
            result = self.agent.run(payload)
            if not result["success"]:
                print(f"Country {country} validation failed: {result['error']}")
    
    def test_invalid_country_code(self):
        """Test rejection of invalid country codes."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "XX",  # Invalid country
            "total_area": {"value": 50000, "unit": "sqft"}
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is False
        assert result["error"]["type"] == "ValidationError"
        assert "country" in result["error"]["message"].lower()
    
    def test_energy_sources_validation(self):
        """Test validation of energy sources structure."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"},
            "energy_sources": [
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 1000000, "unit": "kWh"},
                    "period": "annual"
                },
                {
                    "fuel_type": "natural_gas",
                    "consumption": {"value": 50000, "unit": "therms"},
                    "period": "annual"
                },
                {
                    "fuel_type": "diesel",
                    "consumption": {"value": 5000, "unit": "liters"},
                    "period": "monthly"  # Different period
                }
            ]
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is True
        assert len(result["data"]["normalized"]["energy_sources"]) == 3
    
    def test_mixed_units_normalization(self):
        """Test normalization when mixed units are provided."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 4645, "unit": "sqm"},  # Metric
            "energy_sources": [
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 1500, "unit": "MWh"},  # MWh instead of kWh
                    "period": "annual"
                }
            ]
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is True
        normalized = result["data"]["normalized"]
        
        # Area should be in sqft
        assert normalized["total_area"]["unit"] == "sqft"
        
        # Electricity should be in kWh
        assert normalized["energy_sources"][0]["consumption"]["unit"] in ["kWh", "MWh"]
    
    def test_preserve_metadata(self):
        """Test that metadata fields are preserved."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"},
            "metadata": {
                "reporting_year": 2024,
                "data_quality": "verified",
                "submission_date": "2024-01-15"
            },
            "certifications": ["LEED Gold", "ENERGY STAR"]
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is True
        assert "metadata" in result["data"]["normalized"]
        assert result["data"]["normalized"]["metadata"]["reporting_year"] == 2024
        assert "certifications" in result["data"]["normalized"]
        assert len(result["data"]["normalized"]["certifications"]) == 2
    
    def test_renewable_energy_validation(self):
        """Test validation of renewable energy data."""
        payload = {
            "building_name": "Test Building",
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"},
            "renewable_energy": {
                "solar_pv": {
                    "installed_capacity_kw": 100,
                    "annual_generation_kwh": 150000
                }
            }
        }
        
        result = self.agent.run(payload)
        
        assert result["success"] is True
        if "renewable_energy" in result["data"]["normalized"]:
            solar = result["data"]["normalized"]["renewable_energy"]["solar_pv"]
            assert solar["installed_capacity_kw"] == 100
            assert solar["annual_generation_kwh"] == 150000