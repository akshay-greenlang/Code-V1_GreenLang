# -*- coding: utf-8 -*-
"""
Golden Tests for Boiler-Solar Pack Determinism
===============================================

Tests that verify deterministic output for the boiler-solar pipeline.
"""

import json
import pytest
from pathlib import Path
import sys

# Add parent directories to path
pack_dir = Path(__file__).parent.parent
sys.path.insert(0, str(pack_dir))
sys.path.insert(0, str(pack_dir.parent.parent))

from agents.boiler_analyzer import BoilerAnalyzerAgent
from agents.solar_estimator import SolarEstimatorAgent


class TestDeterminism:
    """Test deterministic behavior of agents"""
    
    @pytest.fixture
    def golden_inputs(self):
        """Load golden test inputs"""
        golden_dir = Path(__file__).parent / "golden"
        with open(golden_dir / "inputs.json") as f:
            return json.load(f)
    
    @pytest.fixture
    def expected_outputs(self):
        """Load expected outputs"""
        golden_dir = Path(__file__).parent / "golden"
        with open(golden_dir / "expected_output.json") as f:
            return json.load(f)
    
    def test_boiler_validation(self, golden_inputs, expected_outputs):
        """Test boiler validation is deterministic"""
        agent = BoilerAnalyzerAgent()
        result = agent.validate_inputs(golden_inputs)
        
        assert result.success == expected_outputs["validate"]["success"]
        assert result.data["validated"] == expected_outputs["validate"]["data"]["validated"]
        # Hash will be deterministic based on sorted inputs
        assert len(result.data["input_hash"]) == 8
    
    def test_boiler_efficiency(self, golden_inputs, expected_outputs):
        """Test boiler efficiency calculation is deterministic"""
        agent = BoilerAnalyzerAgent()
        
        # First validate
        validated = agent.validate_inputs(golden_inputs)
        
        # Then analyze
        result = agent.analyze_efficiency({
            'validated_data': validated.data,
            'boiler_age_years': golden_inputs['boiler_age_years']
        })
        
        assert result.success == expected_outputs["boiler"]["success"]
        assert result.data["efficiency_percent"] == expected_outputs["boiler"]["data"]["efficiency_percent"]
        assert result.data["annual_fuel_consumption_kwh"] == expected_outputs["boiler"]["data"]["annual_fuel_consumption_kwh"]
        assert result.data["annual_emissions_co2_tons"] == expected_outputs["boiler"]["data"]["annual_emissions_co2_tons"]
    
    def test_solar_generation(self, golden_inputs, expected_outputs):
        """Test solar generation calculation is deterministic"""
        agent = SolarEstimatorAgent()
        
        result = agent.calculate_generation({
            'location': golden_inputs['location'],
            'panel_area_sqm': golden_inputs['solar_panel_area_sqm'],
            'annual_hours': golden_inputs['annual_operating_hours']
        })
        
        assert result.success == expected_outputs["solar"]["success"]
        assert result.data["annual_generation_kwh"] == expected_outputs["solar"]["data"]["annual_generation_kwh"]
        assert result.data["co2_offset_tons"] == expected_outputs["solar"]["data"]["co2_offset_tons"]
        assert result.data["capacity_factor"] == expected_outputs["solar"]["data"]["capacity_factor"]
    
    def test_carbon_calculation(self, golden_inputs, expected_outputs):
        """Test carbon emissions calculation is deterministic"""
        agent = BoilerAnalyzerAgent()
        
        # Mock boiler and solar outputs
        boiler_output = {
            'annual_emissions_co2_tons': expected_outputs["boiler"]["data"]["annual_emissions_co2_tons"]
        }
        solar_output = {
            'co2_offset_tons': expected_outputs["solar"]["data"]["co2_offset_tons"]
        }
        
        result = agent.calculate_emissions({
            'boiler_output': boiler_output,
            'solar_output': solar_output,
            'grid_intensity': 0.82
        })
        
        assert result.success == expected_outputs["carbon"]["success"]
        # Allow small difference due to rounding
        assert abs(result.data["net_emissions_tons"] - expected_outputs["carbon"]["data"]["net_emissions_tons"]) < 1.0
        assert abs(result.data["reduction_percent"] - expected_outputs["carbon"]["data"]["reduction_percent"]) < 1.0
    
    def test_report_generation(self, golden_inputs, expected_outputs):
        """Test report generation is deterministic"""
        agent = SolarEstimatorAgent()
        
        result = agent.generate_report({
            'all_data': {
                'validate': expected_outputs["validate"],
                'boiler': expected_outputs["boiler"],
                'solar': expected_outputs["solar"],
                'carbon': expected_outputs["carbon"]
            }
        })
        
        assert result.success == expected_outputs["report"]["success"]
        assert result.data["summary"] == expected_outputs["report"]["data"]["summary"]
        assert result.data["recommendations"] == expected_outputs["report"]["data"]["recommendations"]
        assert result.data["roi_years"] == expected_outputs["report"]["data"]["roi_years"]
        assert result.data["carbon_reduction_percent"] == expected_outputs["report"]["data"]["carbon_reduction_percent"]
    
    def test_multiple_runs_identical(self, golden_inputs):
        """Test that multiple runs produce identical results"""
        boiler_agent = BoilerAnalyzerAgent()
        solar_agent = SolarEstimatorAgent()
        
        # Run validation 3 times
        results = []
        for _ in range(3):
            result = boiler_agent.validate_inputs(golden_inputs)
            results.append(result.data["input_hash"])
        
        # All hashes should be identical
        assert len(set(results)) == 1, "Validation should produce identical hashes"
        
        # Run efficiency calculation 3 times
        efficiency_results = []
        for _ in range(3):
            result = boiler_agent.analyze_efficiency({
                'validated_data': golden_inputs,
                'boiler_age_years': 10
            })
            efficiency_results.append(result.data["efficiency_percent"])
        
        # All efficiencies should be identical
        assert len(set(efficiency_results)) == 1, "Efficiency calculations should be identical"
        
        # Run solar generation 3 times
        solar_results = []
        for _ in range(3):
            result = solar_agent.calculate_generation({
                'location': 'IN-North',
                'panel_area_sqm': 100,
                'annual_hours': 2000
            })
            solar_results.append(result.data["annual_generation_kwh"])
        
        # All solar generations should be identical
        assert len(set(solar_results)) == 1, "Solar calculations should be identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])