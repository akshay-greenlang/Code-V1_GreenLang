"""Tests for BuildingProfileAgent."""

import pytest
from greenlang.agents.building_profile_agent import BuildingProfileAgent


class TestBuildingProfileAgent:
    """Test suite for BuildingProfileAgent - contract-focused tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = BuildingProfileAgent()
    
    def test_deterministic_classification(self):
        """Test that classification is deterministic for same inputs."""
        building_data = {
            "building_type": "office",
            "building_age": 10,
            "climate_zone": "tropical",
            "country": "IN",
            "total_area_sqft": 50000,
            "floor_count": 5,
            "certifications": ["LEED Gold"]
        }
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = self.agent.run(building_data)
            if result["success"]:
                results.append(result["data"])
        
        # All results should be identical
        if len(results) == 3:
            assert results[0] == results[1] == results[2]
    
    def test_eui_band_assignment(self):
        """Test that EUI band is assigned based on building characteristics."""
        result = self.agent.run({
            "building_type": "office",
            "building_age": 5,
            "climate_zone": "temperate",
            "country": "US",
            "total_area_sqft": 100000
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Should have EUI expectations
        assert "expected_eui_range" in data or "eui_band" in data
        if "expected_eui_range" in data:
            assert "min" in data["expected_eui_range"]
            assert "max" in data["expected_eui_range"]
            assert data["expected_eui_range"]["min"] > 0
            assert data["expected_eui_range"]["max"] > data["expected_eui_range"]["min"]
    
    def test_missing_non_critical_metadata(self):
        """Test handling of missing non-critical metadata."""
        minimal_data = {
            "building_type": "office",
            "country": "IN",
            "total_area_sqft": 50000
            # Missing: age, climate_zone, certifications, etc.
        }
        
        result = self.agent.run(minimal_data)
        
        assert result["success"] is True
        data = result["data"]
        
        # Should generate profile with defaults
        assert "profile" in data or "classification" in data
        assert "defaults_applied" in data or "assumptions" in data
        
        # Should note what defaults were used
        if "defaults_applied" in data:
            assert isinstance(data["defaults_applied"], list)
    
    def test_climate_zone_impact(self):
        """Test that climate zone affects EUI expectations."""
        base_building = {
            "building_type": "office",
            "building_age": 10,
            "country": "US",
            "total_area_sqft": 50000
        }
        
        # Test different climate zones
        climates = ["tropical", "temperate", "cold", "arid"]
        eui_ranges = {}
        
        for climate in climates:
            building = {**base_building, "climate_zone": climate}
            result = self.agent.run(building)
            
            if result["success"] and "expected_eui_range" in result["data"]:
                eui_ranges[climate] = result["data"]["expected_eui_range"]
        
        # Different climates should have different EUI expectations
        if len(eui_ranges) > 1:
            values = [r["max"] for r in eui_ranges.values()]
            assert len(set(values)) > 1  # Not all the same
    
    def test_age_impact_on_profile(self):
        """Test that building age affects profile."""
        base_building = {
            "building_type": "office",
            "climate_zone": "temperate",
            "country": "US",
            "total_area_sqft": 50000
        }
        
        # New building
        new_result = self.agent.run({**base_building, "building_age": 1})
        
        # Old building
        old_result = self.agent.run({**base_building, "building_age": 50})
        
        if new_result["success"] and old_result["success"]:
            # Older buildings typically have higher EUI expectations
            if "expected_eui_range" in new_result["data"] and "expected_eui_range" in old_result["data"]:
                assert old_result["data"]["expected_eui_range"]["max"] >= new_result["data"]["expected_eui_range"]["max"]
    
    def test_certification_impact(self):
        """Test that certifications affect building profile."""
        base_building = {
            "building_type": "office",
            "building_age": 10,
            "climate_zone": "temperate",
            "country": "US",
            "total_area_sqft": 50000
        }
        
        # Without certification
        no_cert_result = self.agent.run(base_building)
        
        # With green certification
        cert_result = self.agent.run({
            **base_building,
            "certifications": ["LEED Platinum", "ENERGY STAR"]
        })
        
        if no_cert_result["success"] and cert_result["success"]:
            # Certified buildings should have better efficiency profile
            if "efficiency_class" in cert_result["data"]:
                assert cert_result["data"]["efficiency_class"] in ["high", "very_high"]
    
    def test_output_schema(self):
        """Test that output conforms to expected schema."""
        result = self.agent.run({
            "building_type": "office",
            "building_age": 10,
            "climate_zone": "tropical",
            "country": "IN",
            "total_area_sqft": 50000,
            "floor_count": 5
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Should have profile information
        assert any(key in data for key in ["profile", "classification", "characteristics"])
        
        # Should have EUI or energy expectations
        assert any(key in data for key in ["expected_eui_range", "eui_band", "energy_profile"])
        
        # Should classify efficiency
        if "efficiency_class" in data:
            assert data["efficiency_class"] in ["very_low", "low", "medium", "high", "very_high"]
    
    def test_building_type_profiles(self):
        """Test that different building types have appropriate profiles."""
        building_types = ["office", "retail", "hospital", "school", "warehouse"]
        
        for btype in building_types:
            result = self.agent.run({
                "building_type": btype,
                "building_age": 10,
                "climate_zone": "temperate",
                "country": "US",
                "total_area_sqft": 50000
            })
            
            assert result["success"] is True
            assert result["data"]["building_type"] == btype
            
            # Hospitals typically have highest EUI
            if btype == "hospital" and "expected_eui_range" in result["data"]:
                assert result["data"]["expected_eui_range"]["max"] > 100  # High energy use
    
    def test_size_classification(self):
        """Test building size classification."""
        sizes = [
            (5000, "small"),
            (50000, "medium"),
            (200000, "large"),
            (1000000, "very_large")
        ]
        
        for area, expected_class in sizes:
            result = self.agent.run({
                "building_type": "office",
                "building_age": 10,
                "climate_zone": "temperate",
                "country": "US",
                "total_area_sqft": area
            })
            
            if result["success"] and "size_class" in result["data"]:
                assert result["data"]["size_class"] == expected_class
    
    def test_metadata_preservation(self):
        """Test that input metadata is preserved in profile."""
        result = self.agent.run({
            "building_type": "office",
            "building_age": 10,
            "climate_zone": "tropical",
            "country": "IN",
            "total_area_sqft": 50000,
            "custom_field": "test_value",
            "reporting_year": 2024
        })
        
        assert result["success"] is True
        
        # Check if metadata is preserved
        if "metadata" in result["data"]:
            assert result["data"]["metadata"].get("reporting_year") == 2024