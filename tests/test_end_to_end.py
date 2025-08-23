"""End-to-end workflow tests using actual emission factors from datasets."""

import pytest
from greenlang.core.workflow import Workflow
from greenlang.core.orchestrator import Orchestrator
from greenlang.agents import (
    FuelAgent, CarbonAgent, InputValidatorAgent,
    BenchmarkAgent, GridFactorAgent, BuildingProfileAgent,
    IntensityAgent, RecommendationAgent, ReportAgent
)


class TestEndToEndWorkflows:
    """Test complete workflows with real data."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with all agents."""
        orch = Orchestrator()
        orch.register_agent("validator", InputValidatorAgent())
        orch.register_agent("fuel", FuelAgent())
        orch.register_agent("carbon", CarbonAgent())
        orch.register_agent("benchmark", BenchmarkAgent())
        orch.register_agent("grid_factor", GridFactorAgent())
        orch.register_agent("building_profile", BuildingProfileAgent())
        orch.register_agent("intensity", IntensityAgent())
        orch.register_agent("recommendation", RecommendationAgent())
        orch.register_agent("report", ReportAgent())
        return orch
    
    def test_single_building_minimal_workflow(self, electricity_factors, benchmarks_data):
        """Test minimal workflow for a single building."""
        # Use factors from dataset, not hardcoded
        expected_factor = electricity_factors.get("IN", {}).get("factor", 0.71)
        
        building_data = {
            "metadata": {
                "building_type": "office",
                "area": 50000,
                "area_unit": "sqft",
                "location": {
                    "country": "IN",
                    "city": "Mumbai"
                },
                "occupancy": 200
            },
            "energy_consumption": {
                "electricity": {
                    "value": 1500000,
                    "unit": "kWh"
                }
            }
        }
        
        # Calculate expected emissions using actual factor
        expected_emissions = (1500000 * expected_factor) / 1000  # tons
        
        # Execute workflow
        fuel_agent = FuelAgent()
        result = fuel_agent.execute({
            "fuel_type": "electricity",
            "consumption": 1500000,
            "unit": "kWh",
            "region": "IN"
        })
        
        assert result["success"] is True
        assert abs(result["data"]["co2e_tons"] - expected_emissions) < 0.1
        assert result["data"]["emission_factor"] == expected_factor
    
    def test_multi_fuel_building_workflow(self, emission_factors):
        """Test workflow with multiple fuel types."""
        # Get factors from dataset
        us_factors = emission_factors.get("US", {})
        electricity_factor = us_factors.get("electricity", {}).get("kWh", 0.385)
        gas_factor = us_factors.get("natural_gas", {}).get("therms", 5.3)
        diesel_factor = us_factors.get("diesel", {}).get("gallons", 10.21)
        
        building_data = {
            "metadata": {
                "building_type": "hospital",
                "area": 100000,
                "area_unit": "sqft",
                "location": {"country": "US", "state": "CA"},
                "occupancy": 500
            },
            "energy_consumption": {
                "electricity": {"value": 3500000, "unit": "kWh"},
                "natural_gas": {"value": 50000, "unit": "therms"},
                "diesel": {"value": 5000, "unit": "gallons"}  # Backup generators
            }
        }
        
        # Calculate each fuel's emissions
        fuel_agent = FuelAgent()
        carbon_agent = CarbonAgent()
        
        emissions = []
        
        # Electricity
        elec_result = fuel_agent.execute({
            "fuel_type": "electricity",
            "consumption": 3500000,
            "unit": "kWh",
            "region": "US"
        })
        emissions.append(elec_result["data"])
        
        # Natural gas
        gas_result = fuel_agent.execute({
            "fuel_type": "natural_gas",
            "consumption": 50000,
            "unit": "therms",
            "region": "US"
        })
        emissions.append(gas_result["data"])
        
        # Diesel
        diesel_result = fuel_agent.execute({
            "fuel_type": "diesel",
            "consumption": 5000,
            "unit": "gallons",
            "region": "US"
        })
        emissions.append(diesel_result["data"])
        
        # Aggregate
        total_result = carbon_agent.execute({"emissions": emissions})
        
        assert total_result["success"] is True
        assert total_result["data"]["total_co2e_tons"] > 0
        assert len(total_result["data"]["breakdown"]) == 3
        assert sum(e["percentage"] for e in total_result["data"]["breakdown"]) == pytest.approx(100, rel=0.1)
    
    def test_portfolio_aggregation(self, electricity_factors):
        """Test aggregation across multiple buildings."""
        buildings = [
            {
                "id": "building_1",
                "country": "IN",
                "consumption": 1000000,  # kWh
                "type": "office"
            },
            {
                "id": "building_2", 
                "country": "US",
                "consumption": 2000000,  # kWh
                "type": "retail"
            },
            {
                "id": "building_3",
                "country": "EU",
                "consumption": 1500000,  # kWh
                "type": "warehouse"
            }
        ]
        
        fuel_agent = FuelAgent()
        carbon_agent = CarbonAgent()
        
        all_emissions = []
        
        for building in buildings:
            result = fuel_agent.execute({
                "fuel_type": "electricity",
                "consumption": building["consumption"],
                "unit": "kWh",
                "region": building["country"]
            })
            all_emissions.append(result["data"])
        
        # Aggregate portfolio
        portfolio_result = carbon_agent.execute({"emissions": all_emissions})
        
        assert portfolio_result["success"] is True
        assert portfolio_result["data"]["total_co2e_tons"] > 0
        assert len(portfolio_result["data"]["breakdown"]) == 3
    
    def test_benchmark_with_real_thresholds(self, benchmarks_data):
        """Test benchmarking using actual threshold data."""
        benchmark_agent = BenchmarkAgent()
        
        # Test with actual benchmark thresholds
        result = benchmark_agent.execute({
            "building_type": "office",
            "country": "IN",
            "annual_emissions_tons": 500,
            "area_sqft": 50000
        })
        
        assert result["success"] is True
        assert "rating" in result["data"]
        assert "intensity_kgco2_sqft_year" in result["data"]
        
        # Verify intensity calculation
        expected_intensity = (500 * 1000) / 50000  # kg/sqft/year
        assert abs(result["data"]["intensity_kgco2_sqft_year"] - expected_intensity) < 0.01
    
    def test_intensity_metrics_calculation(self):
        """Test intensity metrics calculation."""
        intensity_agent = IntensityAgent()
        
        result = intensity_agent.execute({
            "total_emissions_tons": 1234.56,
            "metadata": {
                "area": 50000,
                "area_unit": "sqft",
                "occupancy": 200,
                "floor_count": 10
            }
        })
        
        assert result["success"] is True
        assert "intensities" in result["data"]
        assert "per_sqft_year" in result["data"]["intensities"]
        assert "per_person_year" in result["data"]["intensities"]
        assert "per_floor_year" in result["data"]["intensities"]
        
        # Verify calculations
        expected_per_sqft = (1234.56 * 1000) / 50000
        expected_per_person = (1234.56 * 1000) / 200
        expected_per_floor = (1234.56 * 1000) / 10
        
        assert abs(result["data"]["intensities"]["per_sqft_year"] - expected_per_sqft) < 0.01
        assert abs(result["data"]["intensities"]["per_person_year"] - expected_per_person) < 0.01
        assert abs(result["data"]["intensities"]["per_floor_year"] - expected_per_floor) < 0.01
    
    def test_cross_country_factor_verification(self, emission_factors):
        """Verify that different countries have different factors."""
        fuel_agent = FuelAgent()
        consumption = 1000000  # 1M kWh
        
        results = {}
        for country in ["US", "IN", "EU", "CN", "BR"]:
            result = fuel_agent.execute({
                "fuel_type": "electricity",
                "consumption": consumption,
                "unit": "kWh",
                "region": country
            })
            results[country] = result["data"]["co2e_tons"]
        
        # Verify Brazil (cleanest) < EU < US < CN < India (dirtiest)
        assert results["BR"] < results["EU"]
        assert results["EU"] < results["US"]
        assert results["US"] < results["CN"]
        assert results["CN"] < results["IN"]
        
        # Verify actual factors match dataset
        for country in results:
            expected_factor = emission_factors.get(country, {}).get("electricity", {}).get("kWh")
            if expected_factor:
                expected_emissions = (consumption * expected_factor) / 1000
                assert abs(results[country] - expected_emissions) < 0.1
    
    def test_complete_building_analysis_workflow(self, orchestrator):
        """Test complete building analysis workflow."""
        building = {
            "metadata": {
                "building_type": "office",
                "area": 75000,
                "area_unit": "sqft",
                "location": {
                    "country": "US",
                    "state": "NY",
                    "city": "New York"
                },
                "occupancy": 300,
                "floor_count": 15,
                "building_age": 20
            },
            "energy_consumption": {
                "electricity": {"value": 2500000, "unit": "kWh"},
                "natural_gas": {"value": 30000, "unit": "therms"}
            }
        }
        
        # Define workflow steps
        workflow_config = {
            "name": "building_analysis",
            "steps": [
                {"name": "validate", "agent_id": "validator"},
                {"name": "profile", "agent_id": "building_profile"},
                {"name": "calculate_electricity", "agent_id": "fuel"},
                {"name": "calculate_gas", "agent_id": "fuel"},
                {"name": "aggregate", "agent_id": "carbon"},
                {"name": "intensity", "agent_id": "intensity"},
                {"name": "benchmark", "agent_id": "benchmark"},
                {"name": "recommendations", "agent_id": "recommendation"}
            ]
        }
        
        # Since we don't have full workflow implementation,
        # we'll test the key components
        validator = InputValidatorAgent()
        fuel_agent = FuelAgent()
        carbon_agent = CarbonAgent()
        intensity_agent = IntensityAgent()
        benchmark_agent = BenchmarkAgent()
        
        # Step 1: Calculate emissions
        elec_result = fuel_agent.execute({
            "fuel_type": "electricity",
            "consumption": 2500000,
            "unit": "kWh",
            "region": "US"
        })
        
        gas_result = fuel_agent.execute({
            "fuel_type": "natural_gas",
            "consumption": 30000,
            "unit": "therms",
            "region": "US"
        })
        
        # Step 2: Aggregate
        total_result = carbon_agent.execute({
            "emissions": [elec_result["data"], gas_result["data"]]
        })
        
        # Step 3: Calculate intensity
        intensity_result = intensity_agent.execute({
            "total_emissions_tons": total_result["data"]["total_co2e_tons"],
            "metadata": building["metadata"]
        })
        
        # Step 4: Benchmark
        benchmark_result = benchmark_agent.execute({
            "building_type": "office",
            "country": "US",
            "annual_emissions_tons": total_result["data"]["total_co2e_tons"],
            "area_sqft": 75000
        })
        
        # Verify complete workflow
        assert elec_result["success"] is True
        assert gas_result["success"] is True
        assert total_result["success"] is True
        assert intensity_result["success"] is True
        assert benchmark_result["success"] is True
        
        # Verify data flow
        assert total_result["data"]["total_co2e_tons"] > 0
        assert intensity_result["data"]["intensities"]["per_sqft_year"] > 0
        assert benchmark_result["data"]["rating"] in ["A", "B", "C", "D", "E", "F"]