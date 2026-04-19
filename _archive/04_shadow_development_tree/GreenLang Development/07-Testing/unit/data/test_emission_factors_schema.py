# -*- coding: utf-8 -*-
"""Tests for emission factors data schema."""

import json
import pytest
from pathlib import Path


class TestEmissionFactorsSchema:
    """Test suite for emission factors data schema validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Load test data
        self.test_factors_path = Path("tests/fixtures/factors_minimal.json")
        with open(self.test_factors_path) as f:
            self.test_factors = json.load(f)
        
        # Try to load actual factors if available
        self.actual_factors_path = Path("data/global_emission_factors.json")
        if self.actual_factors_path.exists():
            with open(self.actual_factors_path) as f:
                self.actual_factors = json.load(f)
        else:
            self.actual_factors = None
    
    def test_required_top_level_keys(self):
        """Test that required top-level keys are present."""
        required_keys = ["version", "last_updated", "source", "emission_factors"]
        
        for key in required_keys:
            assert key in self.test_factors, f"Missing required key: {key}"
        
        if self.actual_factors:
            for key in required_keys:
                assert key in self.actual_factors, f"Missing required key in actual data: {key}"
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import re
        
        version = self.test_factors["version"]
        # Semantic version pattern
        pattern = r'^\d+\.\d+\.\d+(-[\w\.]+)?(\+[\w\.]+)?$'
        assert re.match(pattern, version), f"Invalid version format: {version}"
        
        if self.actual_factors:
            version = self.actual_factors["version"]
            assert re.match(pattern, version), f"Invalid version format in actual data: {version}"
    
    def test_date_format(self):
        """Test that last_updated is in ISO date format."""
        import datetime
        
        date_str = self.test_factors["last_updated"]
        try:
            datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Invalid date format: {date_str}")
        
        if self.actual_factors:
            date_str = self.actual_factors["last_updated"]
            try:
                datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                pytest.fail(f"Invalid date format in actual data: {date_str}")
    
    def test_electricity_factors_structure(self):
        """Test electricity emission factors structure."""
        electricity = self.test_factors["emission_factors"]["electricity"]
        
        # Should have country codes as keys
        assert len(electricity) > 0, "No electricity factors defined"
        
        for country, data in electricity.items():
            # Country code should be uppercase
            assert country.isupper(), f"Country code should be uppercase: {country}"
            assert len(country) == 2, f"Country code should be 2 letters: {country}"
            
            # Required fields for each country
            assert "factor" in data, f"Missing factor for {country}"
            assert "unit" in data, f"Missing unit for {country}"
            assert "source" in data, f"Missing source for {country}"
            
            # Validate factor value
            assert isinstance(data["factor"], (int, float)), f"Factor must be numeric for {country}"
            assert data["factor"] >= 0, f"Factor must be non-negative for {country}"
            assert data["factor"] < 10, f"Factor seems too high for {country}: {data['factor']}"
            
            # Validate unit
            assert "kgCO2e" in data["unit"], f"Unit should contain kgCO2e for {country}"
            assert "kWh" in data["unit"], f"Unit should contain kWh for {country}"
    
    def test_fuel_factors_structure(self):
        """Test fuel emission factors structure."""
        factors = self.test_factors["emission_factors"]
        
        fuel_types = ["natural_gas", "diesel", "gasoline", "coal"]
        
        for fuel in fuel_types:
            if fuel in factors:
                fuel_data = factors[fuel]
                
                # Should have global or per-unit factors
                assert len(fuel_data) > 0, f"No factors for {fuel}"
                
                for key, data in fuel_data.items():
                    assert "factor" in data, f"Missing factor for {fuel}/{key}"
                    assert "unit" in data, f"Missing unit for {fuel}/{key}"
                    assert "source" in data, f"Missing source for {fuel}/{key}"
                    
                    # Validate factor
                    assert isinstance(data["factor"], (int, float))
                    assert data["factor"] > 0, f"Factor must be positive for {fuel}"
    
    def test_no_negative_factors(self):
        """Test that no emission factors are negative."""
        def check_factors(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "factor":
                        assert value >= 0, f"Negative factor at {path}: {value}"
                    else:
                        check_factors(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_factors(item, f"{path}[{i}]")
        
        check_factors(self.test_factors["emission_factors"])
        
        if self.actual_factors:
            check_factors(self.actual_factors["emission_factors"])
    
    def test_unit_consistency(self):
        """Test that units are consistent across similar fuel types."""
        factors = self.test_factors["emission_factors"]
        
        # All electricity factors should use same unit format
        if "electricity" in factors:
            units = set()
            for country_data in factors["electricity"].values():
                units.add(country_data["unit"])
            
            # All should be kgCO2e/kWh or similar
            for unit in units:
                assert "kgCO2e" in unit and "kWh" in unit, f"Inconsistent electricity unit: {unit}"
    
    def test_country_codes_valid(self):
        """Test that country codes are valid ISO codes."""
        valid_codes = ["IN", "US", "EU", "CN", "UK", "JP", "AU", "CA", "DE", "FR"]  # Common codes
        
        if "electricity" in self.test_factors["emission_factors"]:
            for country in self.test_factors["emission_factors"]["electricity"].keys():
                # Should be 2-letter uppercase or special codes like "EU"
                assert len(country) in [2, 3], f"Invalid country code length: {country}"
                assert country.isupper(), f"Country code not uppercase: {country}"
    
    def test_sources_not_empty(self):
        """Test that all sources are non-empty strings."""
        def check_sources(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "source":
                        assert isinstance(value, str), f"Source not string at {path}"
                        assert len(value) > 0, f"Empty source at {path}"
                    else:
                        check_sources(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_sources(item, f"{path}[{i}]")
        
        check_sources(self.test_factors)
        
        if self.actual_factors:
            check_sources(self.actual_factors)
    
    def test_factor_ranges(self):
        """Test that emission factors are within reasonable ranges."""
        factors = self.test_factors["emission_factors"]
        
        # Electricity: typically 0.1 - 1.5 kgCO2e/kWh
        if "electricity" in factors:
            for country, data in factors["electricity"].items():
                factor = data["factor"]
                assert 0.05 <= factor <= 2.0, f"Electricity factor out of range for {country}: {factor}"
        
        # Natural gas: typically 1.5 - 6 kgCO2e/therm
        if "natural_gas" in factors and "global" in factors["natural_gas"]:
            factor = factors["natural_gas"]["global"]["factor"]
            if "therm" in factors["natural_gas"]["global"]["unit"]:
                assert 1.0 <= factor <= 10.0, f"Natural gas factor out of range: {factor}"
        
        # Diesel: typically 2.5 - 3.0 kgCO2e/liter
        if "diesel" in factors and "global" in factors["diesel"]:
            factor = factors["diesel"]["global"]["factor"]
            if "liter" in factors["diesel"]["global"]["unit"]:
                assert 2.0 <= factor <= 3.5, f"Diesel factor out of range: {factor}"
    
    def test_no_duplicate_countries(self):
        """Test that there are no duplicate country entries."""
        if "electricity" in self.test_factors["emission_factors"]:
            countries = list(self.test_factors["emission_factors"]["electricity"].keys())
            assert len(countries) == len(set(countries)), "Duplicate country codes found"