"""Tests for IntensityAgent."""

import pytest
from greenlang.agents.intensity_agent import IntensityAgent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from test_utils import assert_close


class TestIntensityAgent:
    """Test suite for IntensityAgent - contract-focused tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = IntensityAgent()
    
    def test_per_sqft_intensity(self):
        """Test kgCO2e/sqft/year calculation."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "total_area_sqft": 50000.0,
            "period": "annual"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # 1,000,000 kg / 50,000 sqft = 20 kg/sqft
        assert_close(data["per_sqft"], 20.0, rel_tol=1e-9)
        assert data["unit_per_sqft"] == "kgCO2e/sqft/year"
    
    def test_per_person_intensity(self):
        """Test kgCO2e/person/year calculation."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "occupancy": 250,
            "period": "annual"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # 1,000,000 kg / 250 people = 4,000 kg/person
        assert_close(data["per_person"], 4000.0, rel_tol=1e-9)
        assert data["unit_per_person"] == "kgCO2e/person/year"
    
    def test_per_floor_intensity(self):
        """Test kgCO2e/floor/year calculation."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "floor_count": 10,
            "period": "annual"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # 1,000,000 kg / 10 floors = 100,000 kg/floor
        assert_close(data["per_floor"], 100000.0, rel_tol=1e-9)
        assert data["unit_per_floor"] == "kgCO2e/floor/year"
    
    def test_division_by_zero_area(self):
        """Test handling of zero area."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "total_area_sqft": 0,
            "period": "annual"
        })
        
        assert result["success"] is True  # Partial success
        data = result["data"]
        
        # Should return None or handle gracefully
        assert data.get("per_sqft") is None or data.get("per_sqft") == "N/A"
        assert "reason" in data or "note" in data
        assert "zero" in str(data).lower() or "division" in str(data).lower()
    
    def test_division_by_zero_occupancy(self):
        """Test handling of zero occupancy."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "occupancy": 0,
            "period": "annual"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        assert data.get("per_person") is None or data.get("per_person") == "N/A"
    
    def test_all_intensities_calculated(self):
        """Test calculation of all intensity metrics."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "total_area_sqft": 50000.0,
            "occupancy": 250,
            "floor_count": 10,
            "period": "annual"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # All metrics should be present
        assert "per_sqft" in data
        assert "per_person" in data
        assert "per_floor" in data
        
        # Verify calculations
        assert_close(data["per_sqft"], 20.0, rel_tol=1e-9)
        assert_close(data["per_person"], 4000.0, rel_tol=1e-9)
        assert_close(data["per_floor"], 100000.0, rel_tol=1e-9)
    
    def test_monthly_to_annual_conversion(self):
        """Test conversion from monthly to annual intensities."""
        result = self.agent.run({
            "total_emissions_kg": 83333.33,  # Monthly emissions
            "total_area_sqft": 50000.0,
            "period": "monthly"
        })
        
        if result["success"]:
            data = result["data"]
            
            # If agent converts to annual
            if "annual" in data.get("unit_per_sqft", ""):
                # 83333.33 * 12 / 50000 = 20
                assert_close(data["per_sqft"], 20.0, rel_tol=0.01)
            else:
                # Monthly intensity
                assert_close(data["per_sqft"], 1.667, rel_tol=0.01)
    
    def test_property_scaling_invariance(self):
        """Test that scaling emissions and area by k leaves per-sqft intensity unchanged."""
        base_result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "total_area_sqft": 50000.0,
            "period": "annual"
        })
        
        scaled_result = self.agent.run({
            "total_emissions_kg": 2000000.0,  # 2x emissions
            "total_area_sqft": 100000.0,      # 2x area
            "period": "annual"
        })
        
        assert base_result["success"] and scaled_result["success"]
        
        # Intensity should be the same
        assert_close(
            base_result["data"]["per_sqft"],
            scaled_result["data"]["per_sqft"],
            rel_tol=1e-9
        )
    
    def test_negative_emissions_handling(self):
        """Test handling of negative emissions (net negative building)."""
        result = self.agent.run({
            "total_emissions_kg": -50000.0,  # Net negative
            "total_area_sqft": 50000.0,
            "period": "annual"
        })
        
        if result["success"]:
            # Should calculate negative intensity
            assert result["data"]["per_sqft"] == -1.0
            # Should include note about net negative
            if "note" in result["data"]:
                assert "negative" in result["data"]["note"].lower()
    
    def test_output_schema(self):
        """Test that output conforms to expected schema."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "total_area_sqft": 50000.0,
            "occupancy": 250,
            "period": "annual"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Check for intensity values and their units
        if "per_sqft" in data:
            assert "unit_per_sqft" in data
            assert "kgCO2e" in data["unit_per_sqft"]
            assert "sqft" in data["unit_per_sqft"]
            assert "year" in data["unit_per_sqft"]
        
        if "per_person" in data:
            assert "unit_per_person" in data
            assert "kgCO2e" in data["unit_per_person"]
            assert "person" in data["unit_per_person"]
    
    def test_missing_optional_fields(self):
        """Test handling when optional fields are missing."""
        result = self.agent.run({
            "total_emissions_kg": 1000000.0,
            "total_area_sqft": 50000.0,
            "period": "annual"
            # Missing: occupancy, floor_count
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Should calculate what it can
        assert "per_sqft" in data
        assert_close(data["per_sqft"], 20.0, rel_tol=1e-9)
        
        # Should handle missing fields gracefully
        if "per_person" in data:
            assert data["per_person"] is None or data["per_person"] == "N/A"
    
    def test_very_small_intensities(self):
        """Test calculation with very small intensity values."""
        result = self.agent.run({
            "total_emissions_kg": 1.0,
            "total_area_sqft": 1000000.0,  # Very large area
            "period": "annual"
        })
        
        assert result["success"] is True
        assert_close(result["data"]["per_sqft"], 0.000001, abs_tol=1e-12)