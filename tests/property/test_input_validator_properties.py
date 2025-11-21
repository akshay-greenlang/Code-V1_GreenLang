# -*- coding: utf-8 -*-
"""Property-based tests for InputValidatorAgent."""

from hypothesis import given, strategies as st, assume
import pytest
from greenlang.agents.input_validator_agent import InputValidatorAgent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import assert_close, normalize_factor


class TestInputValidatorProperties:
    """Property tests for InputValidatorAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = InputValidatorAgent()
    
    @given(
        area=st.floats(min_value=0.1, max_value=1e9, allow_nan=False, allow_infinity=False),
        occupancy=st.integers(min_value=0, max_value=100000),
        age=st.integers(min_value=0, max_value=500)
    )
    def test_non_negativity_and_ranges(self, area, occupancy, age):
        """Test that non-negative values are accepted and stay non-negative."""
        building_data = {
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": area, "unit": "sqft"},
            "occupancy": occupancy,
            "building_age": age
        }
        
        result = self.agent.run(building_data)
        
        if result["success"]:
            normalized = result["data"].get("normalized", result["data"])
            
            # Values should remain non-negative
            if "total_area" in normalized:
                assert normalized["total_area"]["value"] >= 0
            if "occupancy" in normalized:
                assert normalized["occupancy"] >= 0
            if "building_age" in normalized:
                assert normalized["building_age"] >= 0
    
    @given(
        negative_area=st.floats(max_value=-0.1, min_value=-1e9, allow_nan=False),
    )
    def test_negative_values_rejected(self, negative_area):
        """Test that negative values are properly rejected."""
        building_data = {
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": negative_area, "unit": "sqft"}
        }
        
        result = self.agent.run(building_data)
        
        # Should fail validation
        assert result["success"] is False
        assert "error" in result
        assert "negative" in result["error"]["message"].lower()
    
    @given(
        sqft_value=st.floats(min_value=1.0, max_value=1e8, allow_nan=False, allow_infinity=False)
    )
    def test_unit_normalization_sqft_sqm_roundtrip(self, sqft_value):
        """Test that unit normalization round-trips correctly."""
        # Test sqft -> sqm -> sqft
        building_sqft = {
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": sqft_value, "unit": "sqft"}
        }
        
        result_sqft = self.agent.run(building_sqft)
        
        if result_sqft["success"]:
            # Convert to sqm
            sqm_value = normalize_factor(sqft_value, "sqft", "sqm")
            
            building_sqm = {
                "building_type": "office",
                "country": "IN",
                "total_area": {"value": sqm_value, "unit": "sqm"}
            }
            
            result_sqm = self.agent.run(building_sqm)
            
            if result_sqm["success"]:
                # Both should normalize to same value (in sqft)
                norm_sqft = result_sqft["data"]["normalized"]["total_area"]["value"]
                norm_sqm = result_sqm["data"]["normalized"]["total_area"]["value"]
                
                # Should be equal within floating point tolerance
                assert_close(norm_sqft, norm_sqm, rel_tol=1e-10)
    
    @given(
        therms=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    def test_gas_unit_roundtrip(self, therms):
        """Test gas unit conversions round-trip correctly."""
        # therms -> m³ -> therms
        m3_value = therms * 2.83168  # Approximate conversion
        
        building_therms = {
            "building_type": "office",
            "country": "US",
            "total_area": {"value": 50000, "unit": "sqft"},
            "energy_sources": [{
                "fuel_type": "natural_gas",
                "consumption": {"value": therms, "unit": "therms"},
                "period": "annual"
            }]
        }
        
        building_m3 = {
            "building_type": "office",
            "country": "EU",
            "total_area": {"value": 50000, "unit": "sqft"},
            "energy_sources": [{
                "fuel_type": "natural_gas",
                "consumption": {"value": m3_value, "unit": "m3"},
                "period": "annual"
            }]
        }
        
        result_therms = self.agent.run(building_therms)
        result_m3 = self.agent.run(building_m3)
        
        if result_therms["success"] and result_m3["success"]:
            # Both should have valid normalized values
            assert "normalized" in result_therms["data"]
            assert "normalized" in result_m3["data"]
    
    def test_unknown_key_helpful_error(self):
        """Test that unknown keys provide helpful error with nearest valid key."""
        # Common typos and their corrections
        typo_cases = [
            ("nat_gas", "natural_gas"),
            ("electricty", "electricity"),
            ("building_typ", "building_type"),
            ("ocupancy", "occupancy"),
            ("total_are", "total_area"),
            ("contry", "country"),
        ]
        
        for typo, correct in typo_cases:
            if typo in ["nat_gas", "electricty"]:
                # Energy source typo
                building_data = {
                    "building_type": "office",
                    "country": "IN",
                    "total_area": {"value": 50000, "unit": "sqft"},
                    "energy_sources": [{
                        "fuel_type": typo,
                        "consumption": {"value": 1000, "unit": "kWh"}
                    }]
                }
            else:
                # Top-level field typo
                building_data = {
                    typo: "office" if "typ" in typo else "IN" if "contry" in typo else 100,
                    "building_type": "office",
                    "country": "IN",
                    "total_area": {"value": 50000, "unit": "sqft"}
                }
                if typo in building_data and typo != correct:
                    del building_data[correct]  # Remove correct key if it exists
            
            result = self.agent.run(building_data)
            
            if not result["success"]:
                error_msg = result["error"]["message"].lower()
                # Should mention the typo
                assert typo in error_msg or "unknown" in error_msg or "invalid" in error_msg
                # Should suggest the correct key
                assert correct in error_msg or "did you mean" in error_msg or "similar" in error_msg
    
    def test_ambiguous_units_error(self):
        """Test helpful errors for ambiguous units."""
        ambiguous_cases = [
            ("gal", ["gallons", "imperial gallons"]),
            ("ton", ["metric tons", "short tons", "long tons"]),
            ("btu", ["BTU", "MMBtu", "therms"]),
        ]
        
        for ambiguous, suggestions in ambiguous_cases:
            building_data = {
                "building_type": "office",
                "country": "US",
                "total_area": {"value": 50000, "unit": "sqft"},
                "energy_sources": [{
                    "fuel_type": "fuel_oil",
                    "consumption": {"value": 1000, "unit": ambiguous}
                }]
            }
            
            result = self.agent.run(building_data)
            
            if not result["success"]:
                error_msg = result["error"]["message"].lower()
                # Should mention ambiguity or suggest alternatives
                assert any(sug.lower() in error_msg for sug in suggestions) or \
                       "ambiguous" in error_msg or "specify" in error_msg
    
    @given(
        building_type=st.sampled_from(["office", "hospital", "retail", "warehouse"]),
        country=st.sampled_from(["IN", "US", "EU", "CN"]),
        area=st.floats(min_value=100, max_value=1e6),
        unit=st.sampled_from(["sqft", "sqm"])
    )
    def test_valid_combinations_accepted(self, building_type, country, area, unit):
        """Test that valid combinations are always accepted."""
        building_data = {
            "building_type": building_type,
            "country": country,
            "total_area": {"value": area, "unit": unit}
        }
        
        result = self.agent.run(building_data)
        
        # Valid data should be accepted
        assert result["success"] is True
        assert "normalized" in result["data"] or "validated" in result["data"]
    
    @given(
        nan_field=st.sampled_from(["area", "occupancy", "consumption"])
    )
    def test_nan_inf_rejected(self, nan_field):
        """Test that NaN and Inf values are rejected."""
        import math
        
        for bad_value in [float('nan'), float('inf'), float('-inf')]:
            if nan_field == "area":
                building_data = {
                    "building_type": "office",
                    "country": "IN",
                    "total_area": {"value": bad_value, "unit": "sqft"}
                }
            elif nan_field == "occupancy":
                building_data = {
                    "building_type": "office",
                    "country": "IN",
                    "total_area": {"value": 50000, "unit": "sqft"},
                    "occupancy": bad_value
                }
            else:  # consumption
                building_data = {
                    "building_type": "office",
                    "country": "IN",
                    "total_area": {"value": 50000, "unit": "sqft"},
                    "energy_sources": [{
                        "fuel_type": "electricity",
                        "consumption": {"value": bad_value, "unit": "kWh"}
                    }]
                }
            
            result = self.agent.run(building_data)
            
            # Should reject non-finite values
            assert result["success"] is False
            error_msg = result["error"]["message"].lower()
            assert any(word in error_msg for word in ["nan", "infinite", "inf", "invalid", "numeric"])