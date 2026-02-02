# -*- coding: utf-8 -*-
"""
Test suite for Climatenza AI agents
"""

import pytest
import json
import pandas as pd
from unittest.mock import Mock, patch, mock_open
import sys
import os

# Removed sys.path manipulation - using installed package

from greenlang.agents import (
    SiteInputAgent, SolarResourceAgent, LoadProfileAgent,
    FieldLayoutAgent, EnergyBalanceAgent
)
from greenlang.agents.base import AgentResult


class TestSiteInputAgent:
    """Test SiteInputAgent functionality"""
    
    def test_load_valid_yaml(self):
        """Test loading valid YAML configuration"""
        yaml_content = """
site:
  name: "Test Site"
  country: "IN"
  lat: 16.506
  lon: 80.648
  tz: "Asia/Kolkata"
process_demand:
  medium: "hot_water"
  temp_in_C: 60
  temp_out_C: 85
  flow_profile: "test.csv"
  schedule: {}
boiler:
  type: "NG"
  efficiency_pct: 84
solar_config:
  tech: "ASC"
  orientation: "N-S"
  row_spacing_factor: 2.2
  tracking: "1-axis"
finance:
  currency: "INR"
  discount_rate_pct: 10
  capex_breakdown: {}
  opex_pct_of_capex: 2
  tariff_fuel_per_kWh: 7.5
  tariff_elec_per_kWh: 9.0
  escalation_fuel_pct: 4
  escalation_elec_pct: 3
assumptions:
  cleaning_cycle_days: 14
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            agent = SiteInputAgent()
            result = agent.execute({"site_file": "test.yaml"})
            
            assert result.success == True
            assert "site" in result.data
            assert result.data["site"]["name"] == "Test Site"
            assert result.data["site"]["lat"] == 16.506
    
    def test_invalid_yaml(self):
        """Test handling of invalid YAML"""
        agent = SiteInputAgent()
        result = agent.execute({"site_file": "nonexistent.yaml"})
        
        assert result.success == False
        assert result.error is not None


class TestSolarResourceAgent:
    """Test SolarResourceAgent functionality"""
    
    def test_generate_solar_data(self):
        """Test solar resource data generation"""
        agent = SolarResourceAgent()
        result = agent.execute({"lat": 16.506, "lon": 80.648})
        
        assert result.success == True
        assert "solar_resource_df" in result.data
        
        # Parse the JSON data
        df_json = result.data["solar_resource_df"]
        df = pd.read_json(df_json, orient="split")
        
        # Check data structure
        assert len(df) == 8760  # Full year hourly data
        assert "dni_w_per_m2" in df.columns
        assert "temp_c" in df.columns
        
        # Check data ranges
        assert df["dni_w_per_m2"].max() <= 1000  # Reasonable DNI max
        assert df["dni_w_per_m2"].min() >= 0
        assert df["temp_c"].max() <= 50  # Reasonable temperature max
        assert df["temp_c"].min() >= 0
    
    def test_missing_coordinates(self):
        """Test handling of missing coordinates"""
        agent = SolarResourceAgent()
        result = agent.execute({})
        
        assert result.success == False
        assert "lat and lon must be provided" in result.error


class TestLoadProfileAgent:
    """Test LoadProfileAgent functionality"""
    
    def test_calculate_load_profile(self):
        """Test load profile calculation"""
        csv_content = """timestamp,flow_kg_s
2024-01-01 00:00:00,2.5
2024-01-01 01:00:00,2.3
2024-01-01 02:00:00,2.1
"""
        
        process_demand = {
            "flow_profile": "test.csv",
            "temp_in_C": 60,
            "temp_out_C": 85
        }
        
        with patch("builtins.open", mock_open(read_data=csv_content)):
            with patch("pandas.read_csv") as mock_read_csv:
                # Create a mock DataFrame
                df = pd.DataFrame({
                    "flow_kg_s": [2.5, 2.3, 2.1]
                })
                mock_read_csv.return_value = df
                
                agent = LoadProfileAgent()
                result = agent.execute({"process_demand": process_demand})
                
                assert result.success == True
                assert "load_profile_df_json" in result.data
                assert "total_annual_demand_gwh" in result.data
                assert result.data["total_annual_demand_gwh"] >= 0
    
    def test_missing_process_demand(self):
        """Test handling of missing process demand"""
        agent = LoadProfileAgent()
        result = agent.execute({})
        
        assert result.success == False
        assert "process_demand not provided" in result.error


class TestFieldLayoutAgent:
    """Test FieldLayoutAgent functionality"""
    
    def test_size_solar_field(self):
        """Test solar field sizing calculation"""
        solar_config = {
            "row_spacing_factor": 2.2
        }
        
        agent = FieldLayoutAgent()
        result = agent.execute({
            "total_annual_demand_gwh": 1.5,
            "solar_config": solar_config
        })
        
        assert result.success == True
        assert "required_aperture_area_m2" in result.data
        assert "num_collectors" in result.data
        assert "required_land_area_m2" in result.data
        
        # Check reasonable values
        assert result.data["required_aperture_area_m2"] > 0
        assert result.data["num_collectors"] > 0
        assert result.data["required_land_area_m2"] > result.data["required_aperture_area_m2"]
    
    def test_missing_inputs(self):
        """Test handling of missing inputs"""
        agent = FieldLayoutAgent()
        result = agent.execute({})
        
        assert result.success == False
        assert "must be provided" in result.error


class TestEnergyBalanceAgent:
    """Test EnergyBalanceAgent functionality"""
    
    def test_energy_balance_simulation(self):
        """Test energy balance simulation"""
        # Create test data
        timestamps = pd.date_range(start="2024-01-01", periods=24, freq="h")
        solar_df = pd.DataFrame({
            "dni_w_per_m2": [0] * 6 + [100, 200, 300, 400, 500, 600, 600, 500, 400, 300, 200, 100] + [0] * 6,
            "temp_c": [15] * 24
        }, index=timestamps)
        
        load_df = pd.DataFrame({
            "demand_kWh": [50] * 24
        }, index=timestamps)
        
        agent = EnergyBalanceAgent()
        result = agent.execute({
            "solar_resource_df_json": solar_df.to_json(orient="split"),
            "load_profile_df_json": load_df.to_json(orient="split"),
            "required_aperture_area_m2": 100
        })
        
        assert result.success == True
        assert "solar_fraction" in result.data
        assert "performance_df_json" in result.data
        assert "total_solar_yield_gwh" in result.data
        
        # Check reasonable values
        assert 0 <= result.data["solar_fraction"] <= 1
        assert result.data["total_solar_yield_gwh"] >= 0
    
    def test_missing_inputs(self):
        """Test handling of missing inputs"""
        agent = EnergyBalanceAgent()
        result = agent.execute({})
        
        assert result.success == False
        assert "Missing required inputs" in result.error


# Integration test
class TestClimatenzaWorkflow:
    """Test complete Climatenza workflow"""
    
    @pytest.mark.integration
    def test_complete_workflow(self):
        """Test complete feasibility analysis workflow"""
        from greenlang.core.orchestrator import Orchestrator
        from greenlang.core.workflow import Workflow
        
        # This would be a full integration test
        # For now, we just check that all agents can be instantiated
        agents = [
            SiteInputAgent(),
            SolarResourceAgent(),
            LoadProfileAgent(),
            FieldLayoutAgent(),
            EnergyBalanceAgent()
        ]
        
        for agent in agents:
            assert agent is not None
            assert hasattr(agent, 'execute')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])