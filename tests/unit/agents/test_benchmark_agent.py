"""Tests for BenchmarkAgent."""

import pytest
from greenlang.agents.benchmark_agent import BenchmarkAgent


class TestBenchmarkAgent:
    """Test suite for BenchmarkAgent - contract-focused tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = BenchmarkAgent()
    
    def test_rating_at_boundaries(self):
        """Test correct rating at exact threshold values."""
        # Test exact boundary values
        test_cases = [
            {"intensity": 5.0, "expected": "A"},   # Excellent
            {"intensity": 10.0, "expected": "B"},  # Good (boundary)
            {"intensity": 15.0, "expected": "C"},  # Average (boundary)
            {"intensity": 20.0, "expected": "D"},  # Below Average (boundary)
            {"intensity": 25.0, "expected": "E"},  # Poor (boundary)
            {"intensity": 30.0, "expected": "F"},  # Very Poor
        ]
        
        for case in test_cases:
            result = self.agent.run({
                "emission_intensity_per_sqft": case["intensity"],
                "building_type": "office",
                "country": "IN"
            })
            
            if result["success"]:
                assert result["data"]["rating"] == case["expected"]
                assert result["data"]["performance_level"] in [
                    "Excellent", "Good", "Average", "Below Average", "Poor", "Very Poor"
                ]
    
    def test_unsupported_building_type(self):
        """Test error handling for unsupported building types."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 15.0,
            "building_type": "spaceship",  # Invalid type
            "country": "IN"
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["ValidationError", "DataNotFoundError"]
        assert "building type" in result["error"]["message"].lower()
        # Should suggest valid types
        assert any(x in result["error"]["message"] for x in ["office", "retail", "hospital"])
    
    def test_unsupported_country(self):
        """Test error handling for unsupported countries."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 15.0,
            "building_type": "office",
            "country": "XX"  # Invalid country
        })
        
        assert result["success"] is False
        assert result["error"]["type"] in ["ValidationError", "DataNotFoundError"]
        assert "country" in result["error"]["message"].lower()
    
    def test_region_specific_benchmarks(self):
        """Test that region-specific benchmarks are applied."""
        # Same intensity, different countries
        intensity = 15.0
        
        result_in = self.agent.run({
            "emission_intensity_per_sqft": intensity,
            "building_type": "office",
            "country": "IN"
        })
        
        result_us = self.agent.run({
            "emission_intensity_per_sqft": intensity,
            "building_type": "office",
            "country": "US"
        })
        
        if result_in["success"] and result_us["success"]:
            # Ratings might differ based on regional benchmarks
            assert "rating" in result_in["data"]
            assert "rating" in result_us["data"]
            assert "benchmark_source" in result_in["data"] or "region" in result_in["data"]
    
    def test_output_schema(self):
        """Test that output conforms to expected schema."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 12.5,
            "building_type": "office",
            "country": "IN"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Required fields
        assert "rating" in data
        assert data["rating"] in ["A", "B", "C", "D", "E", "F"]
        assert "performance_level" in data
        assert "emission_intensity" in data
        assert data["emission_intensity"] == 12.5
        
        # Optional but useful fields
        if "percentile" in data:
            assert 0 <= data["percentile"] <= 100
        if "benchmark_range" in data:
            assert "min" in data["benchmark_range"]
            assert "max" in data["benchmark_range"]
    
    def test_extreme_values(self):
        """Test handling of extreme intensity values."""
        # Very low intensity (should be A)
        result_low = self.agent.run({
            "emission_intensity_per_sqft": 0.1,
            "building_type": "office",
            "country": "IN"
        })
        
        assert result_low["success"] is True
        assert result_low["data"]["rating"] == "A"
        
        # Very high intensity (should be F)
        result_high = self.agent.run({
            "emission_intensity_per_sqft": 100.0,
            "building_type": "office",
            "country": "IN"
        })
        
        assert result_high["success"] is True
        assert result_high["data"]["rating"] == "F"
    
    def test_zero_intensity(self):
        """Test handling of zero intensity (net-zero building)."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 0.0,
            "building_type": "office",
            "country": "IN"
        })
        
        assert result["success"] is True
        assert result["data"]["rating"] == "A"
        if "note" in result["data"]:
            assert "net-zero" in result["data"]["note"].lower() or "zero" in result["data"]["note"].lower()
    
    def test_negative_intensity(self):
        """Test handling of negative intensity (net-positive building)."""
        result = self.agent.run({
            "emission_intensity_per_sqft": -5.0,
            "building_type": "office",
            "country": "IN"
        })
        
        if result["success"]:
            assert result["data"]["rating"] == "A"
            if "note" in result["data"]:
                assert "net-positive" in result["data"]["note"].lower() or "negative" in result["data"]["note"].lower()
    
    @pytest.mark.parametrize("building_type", ["office", "retail", "hospital", "school"])
    def test_different_building_types(self, building_type):
        """Test that different building types have appropriate benchmarks."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 15.0,
            "building_type": building_type,
            "country": "IN"
        })
        
        assert result["success"] is True
        assert result["data"]["rating"] in ["A", "B", "C", "D", "E", "F"]
        assert result["data"]["building_type"] == building_type
    
    def test_comparison_to_average(self):
        """Test comparison to average performance."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 15.0,
            "building_type": "office",
            "country": "IN"
        })
        
        if result["success"] and "comparison_to_average" in result["data"]:
            comparison = result["data"]["comparison_to_average"]
            assert "percentage" in comparison or "factor" in comparison
            assert "direction" in comparison  # "above" or "below"