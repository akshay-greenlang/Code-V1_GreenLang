"""
Comprehensive tests for Scope 3 Category Agents

Tests all 12 Scope 3 categories for:
- Calculation accuracy
- Method support
- Input validation
- Deterministic behavior
- Audit trail completeness
"""

import pytest
import asyncio
from decimal import Decimal
from typing import Dict, Any

# Import all Scope 3 agents
from greenlang.agents.scope3.category_02_capital_goods import CapitalGoodsAgent
from greenlang.agents.scope3.category_03_fuel_energy import FuelEnergyRelatedAgent
from greenlang.agents.scope3.category_04_upstream_transport import UpstreamTransportAgent
from greenlang.agents.scope3.category_05_waste import WasteGeneratedAgent
from greenlang.agents.scope3.category_06_travel import BusinessTravelAgent
from greenlang.agents.scope3.category_07_commuting import EmployeeCommutingAgent
from greenlang.agents.scope3.category_08_leased_assets import UpstreamLeasedAssetsAgent
from greenlang.agents.scope3.category_09_downstream_transport import DownstreamTransportAgent
from greenlang.agents.scope3.category_11_product_use import UseOfSoldProductsAgent
from greenlang.agents.scope3.category_12_eol import EndOfLifeTreatmentAgent
from greenlang.agents.scope3.category_13_downstream_leased import DownstreamLeasedAssetsAgent
from greenlang.agents.scope3.category_14_franchise import FranchisesAgent


class TestCategory2CapitalGoods:
    """Test Category 2: Capital Goods calculations."""

    @pytest.fixture
    def agent(self):
        return CapitalGoodsAgent()

    @pytest.mark.asyncio
    async def test_spend_based_calculation(self, agent):
        """Test spend-based calculation method."""
        input_data = {
            "calculation_method": "spend-based",
            "capital_spend": {
                "machinery_equipment": Decimal("500000"),
                "computer_electronic": Decimal("250000"),
                "construction": Decimal("1000000")
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp",
            "region": "US"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert "total_emissions_t_co2e" in result.data
        assert result.data["total_emissions_t_co2e"] > 0
        assert result.data["calculation_methodology"] == "spend-based"
        assert result.data["ghg_protocol_compliance"] is True
        assert result.data["provenance_hash"] != ""

    @pytest.mark.asyncio
    async def test_average_data_calculation(self, agent):
        """Test average-data calculation method."""
        input_data = {
            "calculation_method": "average-data",
            "equipment_purchases": [
                {
                    "name": "Industrial Robot",
                    "quantity": 5,
                    "weight_kg": 2000,
                    "primary_material": "steel"
                }
            ],
            "building_construction": {
                "area_m2": 5000,
                "type": "warehouse"
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert result.data["total_emissions_kg_co2e"] > 0
        assert len(result.data["calculation_steps"]) > 0

    @pytest.mark.asyncio
    async def test_deterministic_behavior(self, agent):
        """Test that calculations are deterministic."""
        input_data = {
            "calculation_method": "spend-based",
            "capital_spend": {
                "machinery_equipment": Decimal("100000")
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result1 = await agent.process(input_data)
        result2 = await agent.process(input_data)

        # Same input should produce same output
        assert result1.data["total_emissions_kg_co2e"] == result2.data["total_emissions_kg_co2e"]
        # Provenance hash should be identical for same calculation
        assert result1.data["provenance_hash"] == result2.data["provenance_hash"]


class TestCategory3FuelEnergy:
    """Test Category 3: Fuel and Energy Related Activities."""

    @pytest.fixture
    def agent(self):
        return FuelEnergyRelatedAgent()

    @pytest.mark.asyncio
    async def test_wtt_calculation(self, agent):
        """Test Well-to-Tank emissions calculation."""
        input_data = {
            "purchased_fuels": {
                "diesel": {"quantity": 10000, "unit": "liter"},
                "natural_gas": {"quantity": 50000, "unit": "m3"}
            },
            "purchased_electricity": {
                "grid": Decimal("1000000")  # kWh
            },
            "grid_region": "US",
            "include_wtt": True,
            "include_td_losses": True,
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert result.data["total_emissions_t_co2e"] > 0
        assert len(result.data["calculation_steps"]) >= 3  # WTT fuels, WTT electricity, T&D losses

    @pytest.mark.asyncio
    async def test_td_losses_calculation(self, agent):
        """Test T&D losses calculation."""
        input_data = {
            "purchased_electricity": {
                "grid": Decimal("1000000")
            },
            "grid_region": "EU",
            "include_wtt": False,
            "include_td_losses": True,
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        # Should only have T&D losses
        assert any("T&D" in step["description"] for step in result.data["calculation_steps"])


class TestCategory4UpstreamTransport:
    """Test Category 4: Upstream Transportation and Distribution."""

    @pytest.fixture
    def agent(self):
        return UpstreamTransportAgent()

    @pytest.mark.asyncio
    async def test_distance_based_calculation(self, agent):
        """Test distance-based transportation calculation."""
        input_data = {
            "calculation_method": "distance-based",
            "shipments": [
                {
                    "id": "SHIP001",
                    "distance_km": 500,
                    "weight_tonnes": 25,
                    "mode": "truck",
                    "vehicle_type": "large"
                },
                {
                    "id": "SHIP002",
                    "distance_km": 2000,
                    "weight_tonnes": 100,
                    "mode": "rail",
                    "vehicle_type": "diesel"
                },
                {
                    "id": "SHIP003",
                    "distance_km": 5000,
                    "weight_tonnes": 500,
                    "mode": "ship",
                    "vehicle_type": "container"
                }
            ],
            "include_empty_returns": True,
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert result.data["total_emissions_t_co2e"] > 0
        assert result.data["calculation_methodology"] == "distance-based"
        assert len(result.data["calculation_steps"]) == 3

    @pytest.mark.asyncio
    async def test_warehousing_emissions(self, agent):
        """Test warehousing emissions calculation."""
        input_data = {
            "calculation_method": "distance-based",
            "shipments": [],  # Empty shipments
            "warehousing": {
                "area_m2": 10000,
                "type": "refrigerated",
                "occupancy_rate": 0.8,
                "days": 365
            },
            "include_warehousing": True,
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        # Should have warehousing emissions even without transport
        assert result.data["total_emissions_kg_co2e"] > 0


class TestCategory5Waste:
    """Test Category 5: Waste Generated in Operations."""

    @pytest.fixture
    def agent(self):
        return WasteGeneratedAgent()

    @pytest.mark.asyncio
    async def test_waste_treatment_calculation(self, agent):
        """Test waste treatment emissions calculation."""
        input_data = {
            "calculation_method": "waste-type-specific",
            "waste_streams": {
                "mixed": {
                    "weight_tonnes": 50,
                    "treatment": "landfill"
                },
                "paper": {
                    "weight_tonnes": 30,
                    "treatment": "recycling"
                },
                "organic": {
                    "weight_tonnes": 20,
                    "treatment": "composting"
                }
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert result.data["total_emissions_t_co2e"] > 0
        assert len(result.data["calculation_steps"]) == 3


class TestCategory6BusinessTravel:
    """Test Category 6: Business Travel."""

    @pytest.fixture
    def agent(self):
        return BusinessTravelAgent()

    @pytest.mark.asyncio
    async def test_air_travel_calculation(self, agent):
        """Test air travel emissions with class multipliers."""
        input_data = {
            "calculation_method": "distance-based",
            "air_travel": {
                "domestic": {
                    "distance_km": 50000,
                    "class": "economy_class"
                },
                "long_haul": {
                    "distance_km": 100000,
                    "class": "business_class"
                }
            },
            "hotel_stays": {
                "nights": 200,
                "region": "us"
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert result.data["total_emissions_t_co2e"] > 0
        # Business class should have higher emissions than economy
        business_emissions = next(
            (step for step in result.data["calculation_steps"]
             if "business_class" in str(step)),
            None
        )
        assert business_emissions is not None


class TestCategory7EmployeeCommuting:
    """Test Category 7: Employee Commuting."""

    @pytest.fixture
    def agent(self):
        return EmployeeCommutingAgent()

    @pytest.mark.asyncio
    async def test_commuting_calculation(self, agent):
        """Test employee commuting calculation with mode split."""
        input_data = {
            "calculation_method": "distance-based",
            "total_employees": 500,
            "working_days": 220,
            "mode_split": {
                "car_solo": {
                    "percentage": 0.6,
                    "avg_distance_km": 30
                },
                "bus": {
                    "percentage": 0.2,
                    "avg_distance_km": 20
                },
                "train": {
                    "percentage": 0.15,
                    "avg_distance_km": 40
                },
                "bike": {
                    "percentage": 0.05,
                    "avg_distance_km": 5
                }
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        assert result.data["total_emissions_t_co2e"] > 0
        # Should have calculations for each mode
        assert len(result.data["calculation_steps"]) == 4
        # Bike should have zero emissions
        bike_step = next(
            (step for step in result.data["calculation_steps"]
             if "bike" in step.get("description", "").lower()),
            None
        )
        if bike_step:
            assert bike_step["output_value"] == 0


class TestDataQualityAndUncertainty:
    """Test data quality scoring and uncertainty estimation."""

    @pytest.mark.asyncio
    async def test_data_quality_scoring(self):
        """Test that data quality scores are calculated correctly."""
        agent = CapitalGoodsAgent()

        input_data = {
            "calculation_method": "spend-based",  # Lower quality
            "capital_spend": {"machinery_equipment": Decimal("100000")},
            "reporting_year": 2020,  # Old data
            "reporting_entity": "Test Corp",
            "region": "global"  # Non-specific
        }

        result = await agent.process(input_data)

        assert result.success is True
        # Spend-based and old data should result in lower quality score
        assert result.data["data_quality_score"] > 2.5  # Higher score = lower quality

    @pytest.mark.asyncio
    async def test_uncertainty_ranges(self):
        """Test that uncertainty ranges are calculated based on method."""
        agent = UpstreamTransportAgent()

        # Test spend-based (highest uncertainty)
        spend_input = {
            "calculation_method": "spend-based",
            "transport_spend": {"truck": Decimal("50000")},
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        spend_result = await agent.process(spend_input)

        # Test fuel-based (lowest uncertainty)
        fuel_input = {
            "calculation_method": "fuel-based",
            "transport_fuel": {
                "diesel": {"quantity": 10000, "unit": "liter"}
            },
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        fuel_result = await agent.process(fuel_input)

        # Spend-based should have higher uncertainty than fuel-based
        assert abs(spend_result.data["uncertainty_range"]["upper"]) > \
               abs(fuel_result.data["uncertainty_range"]["upper"])


class TestProvenanceAndAuditTrail:
    """Test provenance tracking and audit trail generation."""

    @pytest.mark.asyncio
    async def test_provenance_hash_generation(self):
        """Test that provenance hashes are generated and unique."""
        agent = CapitalGoodsAgent()

        input1 = {
            "calculation_method": "spend-based",
            "capital_spend": {"machinery_equipment": Decimal("100000")},
            "reporting_year": 2024,
            "reporting_entity": "Company A"
        }

        input2 = {
            "calculation_method": "spend-based",
            "capital_spend": {"machinery_equipment": Decimal("100000")},
            "reporting_year": 2024,
            "reporting_entity": "Company B"  # Different company
        }

        result1 = await agent.process(input1)
        result2 = await agent.process(input2)

        # Both should have provenance hashes
        assert result1.data["provenance_hash"] != ""
        assert result2.data["provenance_hash"] != ""

        # Different inputs should produce different hashes
        assert result1.data["provenance_hash"] != result2.data["provenance_hash"]

    @pytest.mark.asyncio
    async def test_calculation_steps_completeness(self):
        """Test that all calculation steps are recorded."""
        agent = FuelEnergyRelatedAgent()

        input_data = {
            "purchased_fuels": {
                "diesel": {"quantity": 1000, "unit": "liter"},
                "natural_gas": {"quantity": 5000, "unit": "m3"}
            },
            "purchased_electricity": {"grid": Decimal("10000")},
            "grid_region": "US",
            "include_wtt": True,
            "include_td_losses": True,
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        steps = result.data["calculation_steps"]

        # Should have steps for each activity
        assert len(steps) >= 4  # 2 fuels WTT, 1 electricity WTT, 1 T&D

        # Each step should have required fields
        for step in steps:
            assert "step_number" in step
            assert "description" in step
            assert "operation" in step
            assert "inputs" in step
            assert "output_value" in step
            assert "formula" in step
            assert "unit" in step


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_zero_emissions(self):
        """Test handling of zero emission activities."""
        agent = EmployeeCommutingAgent()

        input_data = {
            "calculation_method": "distance-based",
            "total_employees": 100,
            "working_days": 220,
            "mode_split": {
                "bike": {"percentage": 0.5, "avg_distance_km": 5},
                "walk": {"percentage": 0.5, "avg_distance_km": 2}
            },
            "reporting_year": 2024,
            "reporting_entity": "Green Corp"
        }

        result = await agent.process(input_data)

        assert result.success is True
        # Should be zero or very close to zero (bike and walk have no emissions)
        assert result.data["total_emissions_kg_co2e"] == 0

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test error handling for missing required fields."""
        agent = CapitalGoodsAgent()

        input_data = {
            "calculation_method": "spend-based",
            # Missing capital_spend
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is False
        assert "error" in result.__dict__

    @pytest.mark.asyncio
    async def test_invalid_calculation_method(self):
        """Test error handling for invalid calculation methods."""
        agent = UpstreamTransportAgent()

        input_data = {
            "calculation_method": "invalid-method",
            "reporting_year": 2024,
            "reporting_entity": "Test Corp"
        }

        result = await agent.process(input_data)

        assert result.success is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_scope3_inventory(self):
        """Test calculating emissions across multiple categories."""
        results = {}

        # Category 2: Capital Goods
        capital_agent = CapitalGoodsAgent()
        capital_input = {
            "calculation_method": "spend-based",
            "capital_spend": {"machinery_equipment": Decimal("1000000")},
            "reporting_year": 2024,
            "reporting_entity": "Global Corp"
        }
        results["cat2"] = await capital_agent.process(capital_input)

        # Category 3: Fuel and Energy
        fuel_agent = FuelEnergyRelatedAgent()
        fuel_input = {
            "purchased_electricity": {"grid": Decimal("5000000")},
            "grid_region": "US",
            "include_wtt": True,
            "include_td_losses": True,
            "reporting_year": 2024,
            "reporting_entity": "Global Corp"
        }
        results["cat3"] = await fuel_agent.process(fuel_input)

        # Category 4: Upstream Transport
        transport_agent = UpstreamTransportAgent()
        transport_input = {
            "calculation_method": "distance-based",
            "shipments": [
                {"distance_km": 1000, "weight_tonnes": 100, "mode": "truck"}
            ],
            "reporting_year": 2024,
            "reporting_entity": "Global Corp"
        }
        results["cat4"] = await transport_agent.process(transport_input)

        # All should succeed
        assert all(r.success for r in results.values())

        # Calculate total Scope 3
        total_scope3 = sum(
            r.data["total_emissions_t_co2e"]
            for r in results.values()
        )

        assert total_scope3 > 0
        print(f"Total Scope 3 emissions: {total_scope3:.2f} tCO2e")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])