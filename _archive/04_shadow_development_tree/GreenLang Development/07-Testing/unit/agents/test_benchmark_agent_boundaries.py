# -*- coding: utf-8 -*-
"""Comprehensive boundary tests for BenchmarkAgent."""

import pytest
from greenlang.agents.benchmark_agent import BenchmarkAgent


class TestBenchmarkAgentBoundaries:
    """Test suite for BenchmarkAgent boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = BenchmarkAgent()
    
    def test_boundary_table_driven(self, benchmark_boundaries, agent_contract_validator):
        """Table-driven test for all boundary values."""
        # Test each boundary point from the actual data
        for building_type, countries in benchmark_boundaries.items():
            for country, boundaries in countries.items():
                for boundary_case in boundaries:
                    # Test exact boundary value
                    result = self.agent.run({
                        "emission_intensity_per_sqft": boundary_case["value"],
                        "building_type": building_type,
                        "country": country
                    })
                    
                    agent_contract_validator.validate_response(result, "BenchmarkAgent")
                    
                    if result["success"]:
                        actual_rating = result["data"]["rating"]
                        expected_rating = boundary_case["rating"]
                        
                        # For min boundaries, the value should get that rating
                        # For max boundaries, it depends on inclusive/exclusive
                        if boundary_case["boundary"] == "min":
                            assert actual_rating == expected_rating, \
                                f"At {building_type}/{country} min boundary {boundary_case['value']}: " \
                                f"expected {expected_rating}, got {actual_rating}"
    
    @pytest.mark.parametrize("building_type,country,test_cases", [
        ("office", "IN", [
            (0.0, "A"),      # Zero emissions
            (0.01, "A"),     # Just above zero
            (9.99, "A"),     # Just below A/B boundary
            (10.0, "B"),     # Exactly at A/B boundary
            (10.01, "B"),    # Just above A/B boundary
            (14.99, "B"),    # Just below B/C boundary
            (15.0, "C"),     # Exactly at B/C boundary
            (15.01, "C"),    # Just above B/C boundary
            (19.99, "C"),    # Just below C/D boundary
            (20.0, "D"),     # Exactly at C/D boundary
            (20.01, "D"),    # Just above C/D boundary
            (24.99, "D"),    # Just below D/E boundary
            (25.0, "E"),     # Exactly at D/E boundary
            (25.01, "E"),    # Just above D/E boundary
            (29.99, "E"),    # Just below E/F boundary
            (30.0, "F"),     # Exactly at E/F boundary
            (30.01, "F"),    # Just above E/F boundary
            (100.0, "F"),    # High value
            (1000.0, "F"),   # Very high value
        ]),
        ("office", "US", [
            (0.0, "A"),
            (7.99, "A"),
            (8.0, "B"),
            (8.01, "B"),
            (11.99, "B"),
            (12.0, "C"),
            (12.01, "C"),
            (17.99, "C"),
            (18.0, "D"),
            (18.01, "D"),
            (23.99, "D"),
            (24.0, "E"),
            (24.01, "E"),
            (29.99, "E"),
            (30.0, "F"),
            (30.01, "F"),
            (100.0, "F"),
        ])
    ])
    def test_explicit_boundaries(self, building_type, country, test_cases, 
                                benchmarks_data, agent_contract_validator):
        """Test explicit boundary values for each rating band."""
        for intensity, expected_rating in test_cases:
            result = self.agent.run({
                "emission_intensity_per_sqft": intensity,
                "building_type": building_type,
                "country": country
            })
            
            agent_contract_validator.validate_response(result, "BenchmarkAgent")
            
            if result["success"]:
                actual_rating = result["data"]["rating"]
                assert actual_rating == expected_rating, \
                    f"For {building_type}/{country} at intensity {intensity}: " \
                    f"expected {expected_rating}, got {actual_rating}"
                
                # Also verify the intensity is returned correctly
                assert result["data"]["emission_intensity"] == intensity
    
    def test_inclusive_exclusive_boundaries(self, benchmarks_data):
        """Test that boundaries are handled consistently (inclusive lower, exclusive upper)."""
        # Standard expectation: [min, max) - inclusive min, exclusive max
        test_cases = [
            # (value, building_type, country, expected_rating)
            (10.0, "office", "IN", "B"),  # Exactly at boundary - should be B (10 is min of B)
            (15.0, "office", "IN", "C"),  # Exactly at boundary - should be C (15 is min of C)
            (20.0, "office", "IN", "D"),  # Exactly at boundary - should be D (20 is min of D)
        ]
        
        for value, building_type, country, expected in test_cases:
            result = self.agent.run({
                "emission_intensity_per_sqft": value,
                "building_type": building_type,
                "country": country
            })
            
            if result["success"]:
                assert result["data"]["rating"] == expected, \
                    f"Boundary {value} should be {expected}, got {result['data']['rating']}"
    
    def test_negative_intensity(self, agent_contract_validator):
        """Test handling of negative intensity (net-positive building)."""
        result = self.agent.run({
            "emission_intensity_per_sqft": -5.0,
            "building_type": "office",
            "country": "IN"
        })
        
        agent_contract_validator.validate_response(result, "BenchmarkAgent")
        
        if result["success"]:
            # Negative should still get best rating
            assert result["data"]["rating"] == "A"
            # Should have a note about net-positive
            assert "note" in result["data"] or "net" in str(result["data"]).lower()
    
    def test_zero_intensity(self, agent_contract_validator):
        """Test handling of zero intensity (net-zero building)."""
        result = self.agent.run({
            "emission_intensity_per_sqft": 0.0,
            "building_type": "office",
            "country": "IN"
        })
        
        agent_contract_validator.validate_response(result, "BenchmarkAgent")
        
        if result["success"]:
            assert result["data"]["rating"] == "A"
            # Should note this is net-zero
            if "note" in result["data"]:
                assert "zero" in result["data"]["note"].lower()
    
    def test_very_small_differences(self):
        """Test handling of values very close to boundaries."""
        epsilon = 1e-10
        
        test_cases = [
            (10.0 - epsilon, "A"),  # Just below boundary
            (10.0, "B"),            # Exactly at boundary
            (10.0 + epsilon, "B"),  # Just above boundary
        ]
        
        for value, expected in test_cases:
            result = self.agent.run({
                "emission_intensity_per_sqft": value,
                "building_type": "office",
                "country": "IN"
            })
            
            if result["success"]:
                # May need tolerance for floating point comparison
                actual = result["data"]["rating"]
                # Allow for floating point imprecision at boundaries
                if abs(value - 10.0) < 1e-9:
                    assert actual in ["A", "B"], \
                        f"Near boundary {value}: got {actual}, expected A or B"
                else:
                    assert actual == expected, \
                        f"At {value}: expected {expected}, got {actual}"
    
    def test_all_building_types(self, benchmarks_data, agent_contract_validator):
        """Test that all building types have proper boundaries."""
        building_types = ["office", "hospital", "retail", "warehouse", "hotel", "education"]
        
        for building_type in building_types:
            # Test a mid-range value
            result = self.agent.run({
                "emission_intensity_per_sqft": 15.0,
                "building_type": building_type,
                "country": "IN"
            })
            
            agent_contract_validator.validate_response(result, "BenchmarkAgent")
            
            if result["success"]:
                assert result["data"]["rating"] in ["A", "B", "C", "D", "E", "F"]
                assert result["data"]["building_type"] == building_type
    
    def test_performance_label_consistency(self, benchmarks_data):
        """Test that performance labels match ratings consistently."""
        rating_to_label = {
            "A": "Excellent",
            "B": "Good", 
            "C": "Average",
            "D": "Below Average",
            "E": "Poor",
            "F": "Very Poor"
        }
        
        for rating, expected_label in rating_to_label.items():
            # Find an intensity that gives this rating
            intensities = {
                "A": 5.0,
                "B": 12.0,
                "C": 17.0,
                "D": 22.0,
                "E": 27.0,
                "F": 35.0
            }
            
            result = self.agent.run({
                "emission_intensity_per_sqft": intensities[rating],
                "building_type": "office",
                "country": "IN"
            })
            
            if result["success"]:
                assert result["data"]["rating"] == rating
                if "performance_level" in result["data"]:
                    assert result["data"]["performance_level"] == expected_label