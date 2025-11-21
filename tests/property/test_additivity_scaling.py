# -*- coding: utf-8 -*-
"""Property-based tests for additivity and scaling properties."""

from hypothesis import given, strategies as st, assume
from greenlang.agents.fuel_agent import FuelAgent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import assert_close


class TestAdditivityProperty:
    """Test additivity properties of emissions calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = FuelAgent()
    
    @given(
        total=st.floats(min_value=1.0, max_value=1e7, allow_nan=False, allow_infinity=False),
        parts=st.integers(min_value=2, max_value=10)
    )
    def test_additivity(self, total, parts):
        """Test that emissions are additive when split into parts."""
        chunk = total / parts
        
        # Calculate emissions for total
        single_result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": total, "unit": "kWh"},
            "country": "IN"
        })
        
        # Skip if calculation failed
        assume(single_result["success"])
        single_emissions = single_result["data"]["co2e_emissions_kg"]
        
        # Calculate emissions for parts
        summed_emissions = 0.0
        for _ in range(parts):
            part_result = self.agent.run({
                "fuel_type": "electricity",
                "consumption": {"value": chunk, "unit": "kWh"},
                "country": "IN"
            })
            assume(part_result["success"])
            summed_emissions += part_result["data"]["co2e_emissions_kg"]
        
        # Assert additivity within tolerance
        assert_close(single_emissions, summed_emissions, rel_tol=1e-6)
    
    @given(
        base_value=st.floats(min_value=10.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        scale_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    def test_proportionality(self, base_value, scale_factor):
        """Test that emissions scale proportionally with consumption."""
        # Base calculation
        base_result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": base_value, "unit": "kWh"},
            "country": "US"
        })
        
        assume(base_result["success"])
        base_emissions = base_result["data"]["co2e_emissions_kg"]
        
        # Scaled calculation
        scaled_result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": base_value * scale_factor, "unit": "kWh"},
            "country": "US"
        })
        
        assume(scaled_result["success"])
        scaled_emissions = scaled_result["data"]["co2e_emissions_kg"]
        
        # Emissions should scale proportionally
        expected_scaled = base_emissions * scale_factor
        assert_close(scaled_emissions, expected_scaled, rel_tol=1e-9)
    
    @given(
        value=st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
        country=st.sampled_from(["IN", "US", "EU", "CN"])
    )
    def test_non_negativity(self, value, country):
        """Test that valid inputs never yield negative emissions."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": value, "unit": "kWh"},
            "country": country
        })
        
        if result["success"]:
            assert result["data"]["co2e_emissions_kg"] >= 0
    
    @given(
        values=st.lists(
            st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5
        )
    )
    def test_sum_equals_total(self, values):
        """Test that sum of individual calculations equals total calculation."""
        total = sum(values)
        
        # Calculate total emissions
        total_result = self.agent.run({
            "fuel_type": "natural_gas",
            "consumption": {"value": total, "unit": "therms"},
            "country": "US"
        })
        
        assume(total_result["success"])
        total_emissions = total_result["data"]["co2e_emissions_kg"]
        
        # Calculate individual emissions
        individual_emissions = []
        for value in values:
            result = self.agent.run({
                "fuel_type": "natural_gas",
                "consumption": {"value": value, "unit": "therms"},
                "country": "US"
            })
            assume(result["success"])
            individual_emissions.append(result["data"]["co2e_emissions_kg"])
        
        sum_individual = sum(individual_emissions)
        
        # Assert sum equals total
        assert_close(total_emissions, sum_individual, rel_tol=1e-6)