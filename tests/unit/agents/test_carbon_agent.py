"""Tests for CarbonAgent."""

import pytest
from greenlang.agents.carbon_agent import CarbonAgent
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from test_utils import assert_close, assert_percentage_sum


class TestCarbonAgent:
    """Test suite for CarbonAgent - contract-focused tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = CarbonAgent()
    
    def test_sum_of_emissions(self):
        """Test that sum of per-fuel emissions equals total."""
        fuel_emissions = [
            {
                "fuel_type": "electricity",
                "co2e_emissions_kg": 1065000.0,
                "percentage": None
            },
            {
                "fuel_type": "diesel",
                "co2e_emissions_kg": 13400.0,
                "percentage": None
            },
            {
                "fuel_type": "natural_gas",
                "co2e_emissions_kg": 265000.0,
                "percentage": None
            }
        ]
        
        result = self.agent.run({"fuel_emissions": fuel_emissions})
        
        assert result["success"] is True
        data = result["data"]
        
        # Total should equal sum of individual emissions
        expected_total = sum(f["co2e_emissions_kg"] for f in fuel_emissions)
        assert_close(data["total_emissions_kg"], expected_total, rel_tol=1e-9)
        
        # Check breakdown exists
        assert "breakdown" in data
        assert len(data["breakdown"]) == 3
    
    def test_percentage_breakdown(self, agent_contract_validator):
        """Test that percentage breakdown sums to 100%."""
        fuel_emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 750000.0},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 250000.0}
        ]
        
        result = self.agent.run({"fuel_emissions": fuel_emissions})
        
        agent_contract_validator.validate_response(result, "CarbonAgent")
        assert result["success"] is True
        data = result["data"]
        
        # Extract percentages
        percentages = [item["percentage"] for item in data["breakdown"]]
        
        # Percentages must sum to 100% within epsilon
        total_percentage = sum(percentages)
        epsilon = 0.01  # Allow 0.01% tolerance for floating point
        assert abs(total_percentage - 100.0) <= epsilon, \
            f"Percentages sum to {total_percentage}, not 100.0±{epsilon}"
        
        # Verify individual percentages
        assert_close(data["breakdown"][0]["percentage"], 75.0, abs_tol=0.01)
        assert_close(data["breakdown"][1]["percentage"], 25.0, abs_tol=0.01)
        
        # Each percentage must be non-negative
        for item in data["breakdown"]:
            assert item["percentage"] >= 0, f"Negative percentage: {item['percentage']}"
            assert item["percentage"] <= 100, f"Percentage > 100: {item['percentage']}"
    
    def test_empty_input(self, agent_contract_validator):
        """Test handling of empty input returns 0 with reason."""
        result = self.agent.run({"fuel_emissions": []})
        
        agent_contract_validator.validate_response(result, "CarbonAgent")
        assert result["success"] is True
        assert result["data"]["total_emissions_kg"] == 0
        
        # Must provide a reason/note for zero emissions
        assert any(key in result["data"] for key in ["note", "message", "reason"]), \
            "Empty input should include explanation"
        
        # Reason should mention no emissions or empty
        result_str = str(result["data"]).lower()
        assert any(word in result_str for word in ["no emissions", "empty", "zero"]), \
            "Reason should explain zero emissions"
    
    def test_missing_expected_fields(self):
        """Test error handling for missing expected fields."""
        result = self.agent.run({})  # Missing fuel_emissions
        
        assert result["success"] is False
        assert result["error"]["type"] in ["ValidationError", "MissingFieldError"]
        assert "fuel_emissions" in result["error"]["message"]
    
    def test_duplicate_fuel_aggregation(self):
        """Test that duplicate fuel types are aggregated correctly."""
        fuel_emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 500000.0},
            {"fuel_type": "electricity", "co2e_emissions_kg": 300000.0},  # Duplicate
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 200000.0}
        ]
        
        result = self.agent.run({"fuel_emissions": fuel_emissions})
        
        assert result["success"] is True
        data = result["data"]
        
        # Total should include all emissions
        assert_close(data["total_emissions_kg"], 1000000.0, rel_tol=1e-9)
        
        # Breakdown should aggregate electricity
        electricity_entries = [b for b in data["breakdown"] if b["fuel_type"] == "electricity"]
        if len(electricity_entries) == 1:
            # If aggregated into single entry
            assert_close(electricity_entries[0]["co2e_emissions_kg"], 800000.0, rel_tol=1e-9)
    
    def test_output_schema(self):
        """Test that output conforms to expected schema."""
        fuel_emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 1000000.0}
        ]
        
        result = self.agent.run({"fuel_emissions": fuel_emissions})
        
        assert result["success"] is True
        data = result["data"]
        
        # Required fields
        assert "total_emissions_kg" in data
        assert "breakdown" in data
        assert "unit" in data
        assert data["unit"] == "kgCO2e"
        
        # Breakdown structure
        for item in data["breakdown"]:
            assert "fuel_type" in item
            assert "co2e_emissions_kg" in item
            assert "percentage" in item
    
    def test_negative_emissions_handling(self):
        """Test handling of negative emissions (e.g., renewable offsets)."""
        fuel_emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 1000000.0},
            {"fuel_type": "solar_pv", "co2e_emissions_kg": -100000.0}  # Offset
        ]
        
        result = self.agent.run({"fuel_emissions": fuel_emissions})
        
        if result["success"]:
            # Net emissions should be 900000
            assert_close(result["data"]["total_emissions_kg"], 900000.0, rel_tol=1e-9)
            
            # Should track offsets separately if supported
            if "offsets" in result["data"]:
                assert result["data"]["offsets"]["solar_pv"] == 100000.0
    
    def test_very_small_emissions(self):
        """Test handling of very small emission values."""
        fuel_emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 0.001},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 0.002}
        ]
        
        result = self.agent.run({"fuel_emissions": fuel_emissions})
        
        assert result["success"] is True
        assert_close(result["data"]["total_emissions_kg"], 0.003, abs_tol=1e-12)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved through aggregation."""
        fuel_emissions = [
            {
                "fuel_type": "electricity",
                "co2e_emissions_kg": 1000000.0,
                "source": "Grid",
                "period": "annual"
            }
        ]
        
        result = self.agent.run({
            "fuel_emissions": fuel_emissions,
            "reporting_year": 2024,
            "building_id": "test-123"
        })
        
        assert result["success"] is True
        
        # Check if metadata is preserved
        if "metadata" in result["data"]:
            assert result["data"]["metadata"].get("reporting_year") == 2024
            assert result["data"]["metadata"].get("building_id") == "test-123"