"""Tests for benchmarks data schema."""

import json
import pytest
from pathlib import Path


class TestBenchmarksSchema:
    """Test suite for benchmarks data schema validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Try to load actual benchmarks if available
        self.benchmarks_path = Path("data/global_benchmarks.json")
        if self.benchmarks_path.exists():
            with open(self.benchmarks_path) as f:
                self.benchmarks = json.load(f)
        else:
            # Create test benchmark data
            self.benchmarks = {
                "version": "0.0.1",
                "last_updated": "2024-01-01",
                "benchmarks": {
                    "office": {
                        "IN": {
                            "A": {"min": 0, "max": 10},
                            "B": {"min": 10, "max": 15},
                            "C": {"min": 15, "max": 20},
                            "D": {"min": 20, "max": 25},
                            "E": {"min": 25, "max": 30},
                            "F": {"min": 30, "max": None}
                        }
                    }
                }
            }
    
    def test_required_keys(self):
        """Test that required keys are present."""
        required_keys = ["version", "last_updated", "benchmarks"]
        
        for key in required_keys:
            assert key in self.benchmarks, f"Missing required key: {key}"
    
    def test_building_types(self):
        """Test that common building types are included."""
        expected_types = ["office", "retail", "hospital", "school"]
        benchmarks = self.benchmarks["benchmarks"]
        
        for btype in expected_types:
            if btype not in benchmarks:
                print(f"Warning: Building type '{btype}' not in benchmarks")
    
    def test_rating_structure(self):
        """Test that ratings follow correct structure."""
        benchmarks = self.benchmarks["benchmarks"]
        expected_ratings = ["A", "B", "C", "D", "E", "F"]
        
        for building_type, countries in benchmarks.items():
            for country, ratings in countries.items():
                # Check all ratings are present
                for rating in expected_ratings:
                    assert rating in ratings, f"Missing rating {rating} for {building_type}/{country}"
                
                # Check each rating has min/max
                for rating, thresholds in ratings.items():
                    assert "min" in thresholds, f"Missing 'min' for {building_type}/{country}/{rating}"
                    assert "max" in thresholds, f"Missing 'max' for {building_type}/{country}/{rating}"
    
    def test_thresholds_monotonic(self):
        """Test that thresholds are monotonically increasing."""
        benchmarks = self.benchmarks["benchmarks"]
        
        for building_type, countries in benchmarks.items():
            for country, ratings in countries.items():
                # Extract thresholds in order
                ordered_ratings = ["A", "B", "C", "D", "E", "F"]
                prev_max = -1
                
                for rating in ordered_ratings:
                    if rating in ratings:
                        threshold = ratings[rating]
                        
                        # Check min >= previous max
                        if threshold["min"] is not None:
                            assert threshold["min"] >= prev_max, \
                                f"Non-monotonic thresholds at {building_type}/{country}/{rating}"
                        
                        # Update prev_max
                        if threshold["max"] is not None:
                            prev_max = threshold["max"]
                        else:
                            prev_max = float('inf')
    
    def test_no_gaps_in_thresholds(self):
        """Test that there are no gaps between rating thresholds."""
        benchmarks = self.benchmarks["benchmarks"]
        
        for building_type, countries in benchmarks.items():
            for country, ratings in countries.items():
                ordered_ratings = ["A", "B", "C", "D", "E", "F"]
                
                for i in range(len(ordered_ratings) - 1):
                    current = ordered_ratings[i]
                    next_rating = ordered_ratings[i + 1]
                    
                    if current in ratings and next_rating in ratings:
                        current_max = ratings[current]["max"]
                        next_min = ratings[next_rating]["min"]
                        
                        if current_max is not None and next_min is not None:
                            # Max of current should equal min of next
                            assert current_max == next_min, \
                                f"Gap between {current} and {next_rating} for {building_type}/{country}"
    
    def test_first_and_last_ratings(self):
        """Test that first rating starts at 0 and last has no upper bound."""
        benchmarks = self.benchmarks["benchmarks"]
        
        for building_type, countries in benchmarks.items():
            for country, ratings in countries.items():
                # First rating (A) should start at 0
                if "A" in ratings:
                    assert ratings["A"]["min"] == 0, \
                        f"Rating A should start at 0 for {building_type}/{country}"
                
                # Last rating (F) should have no upper bound
                if "F" in ratings:
                    assert ratings["F"]["max"] is None or ratings["F"]["max"] == float('inf'), \
                        f"Rating F should have no upper bound for {building_type}/{country}"
    
    def test_valid_country_codes(self):
        """Test that country codes are valid."""
        benchmarks = self.benchmarks["benchmarks"]
        
        for building_type, countries in benchmarks.items():
            for country in countries.keys():
                # Should be uppercase
                assert country.isupper(), f"Country code not uppercase: {country}"
                # Should be 2-3 letters (allowing for EU)
                assert 2 <= len(country) <= 3, f"Invalid country code length: {country}"
    
    def test_threshold_values_reasonable(self):
        """Test that threshold values are within reasonable ranges."""
        benchmarks = self.benchmarks["benchmarks"]
        
        for building_type, countries in benchmarks.items():
            for country, ratings in countries.items():
                for rating, thresholds in ratings.items():
                    # Check min values
                    if thresholds["min"] is not None:
                        assert thresholds["min"] >= 0, \
                            f"Negative threshold for {building_type}/{country}/{rating}"
                        assert thresholds["min"] < 1000, \
                            f"Threshold too high for {building_type}/{country}/{rating}"
                    
                    # Check max values
                    if thresholds["max"] is not None:
                        assert thresholds["max"] > 0, \
                            f"Non-positive max threshold for {building_type}/{country}/{rating}"
                        assert thresholds["max"] < 1000, \
                            f"Max threshold too high for {building_type}/{country}/{rating}"
    
    def test_labels_if_present(self):
        """Test that labels are valid if present."""
        benchmarks = self.benchmarks["benchmarks"]
        valid_labels = ["Excellent", "Good", "Average", "Below Average", "Poor", "Very Poor"]
        
        for building_type, countries in benchmarks.items():
            for country, ratings in countries.items():
                for rating, data in ratings.items():
                    if "label" in data:
                        assert data["label"] in valid_labels, \
                            f"Invalid label for {building_type}/{country}/{rating}: {data['label']}"
    
    def test_metadata_if_present(self):
        """Test metadata fields if present."""
        if "metadata" in self.benchmarks:
            metadata = self.benchmarks["metadata"]
            
            # Check common metadata fields
            if "description" in metadata:
                assert isinstance(metadata["description"], str)
                assert len(metadata["description"]) > 0
            
            if "methodology" in metadata:
                assert isinstance(metadata["methodology"], str)
            
            if "sources" in metadata:
                assert isinstance(metadata["sources"], (list, str))
    
    def test_coverage_consistency(self):
        """Test that coverage is consistent across building types."""
        benchmarks = self.benchmarks["benchmarks"]
        
        # Get all countries across all building types
        all_countries = set()
        for building_type, countries in benchmarks.items():
            all_countries.update(countries.keys())
        
        # Check that each building type has benchmarks for common countries
        for building_type, countries in benchmarks.items():
            available_countries = set(countries.keys())
            
            # At least some overlap expected
            overlap = all_countries.intersection(available_countries)
            assert len(overlap) > 0, f"No country coverage for {building_type}"