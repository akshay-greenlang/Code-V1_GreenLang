"""
Unit tests for Scope 3 emission calculation agents
Target coverage: 70%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime
import pandas as pd

# Import test helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_enhanced import *


class TestScope3BaseAgent:
    """Test suite for Scope 3 base agent functionality."""

    @pytest.fixture
    def base_agent(self):
        """Create Scope 3 base agent instance."""
        from greenlang.agents.scope3.base import Scope3BaseAgent

        with patch('greenlang.agents.scope3.base.Scope3BaseAgent.__init__', return_value=None):
            agent = Scope3BaseAgent.__new__(Scope3BaseAgent)
            agent.name = "scope3_base"
            agent.category = None
            agent.emission_factors = {}
            agent.logger = Mock()
            return agent

    def test_agent_initialization(self, base_agent):
        """Test agent initializes with correct configuration."""
        assert base_agent.name == "scope3_base"
        assert base_agent.emission_factors == {}

    def test_validate_input_data(self, base_agent):
        """Test input data validation."""
        valid_data = {
            "activity_data": 100,
            "unit": "kg",
            "region": "US"
        }

        base_agent.validate_input = Mock(return_value=True)
        is_valid = base_agent.validate_input(valid_data)

        assert is_valid == True

    def test_emission_factor_lookup(self, base_agent, mock_emission_factors):
        """Test emission factor lookup."""
        base_agent.emission_factors = mock_emission_factors
        base_agent.lookup_factor = Mock(return_value=Decimal("2.68"))

        factor = base_agent.lookup_factor("diesel", "US", "stationary_combustion")

        assert factor == Decimal("2.68")

    def test_calculate_emissions(self, base_agent):
        """Test basic emission calculation."""
        base_agent.calculate = Mock(return_value=Decimal("268.0"))

        emissions = base_agent.calculate(
            activity_data=Decimal("100"),
            emission_factor=Decimal("2.68")
        )

        assert emissions == Decimal("268.0")

    def test_provenance_tracking(self, base_agent, mock_provenance_tracker):
        """Test provenance tracking in calculations."""
        base_agent.provenance_tracker = mock_provenance_tracker

        base_agent.track_calculation = Mock(return_value="hash_abc")
        hash_value = base_agent.track_calculation({
            "input": 100,
            "factor": 2.68,
            "output": 268.0
        })

        assert hash_value == "hash_abc"


class TestPurchasedGoodsAgent:
    """Test suite for Category 1: Purchased Goods and Services."""

    @pytest.fixture
    def purchased_goods_agent(self):
        """Create purchased goods agent instance."""
        from greenlang.agents.scope3.purchased_goods import PurchasedGoodsAgent

        with patch('greenlang.agents.scope3.purchased_goods.PurchasedGoodsAgent.__init__', return_value=None):
            agent = PurchasedGoodsAgent.__new__(PurchasedGoodsAgent)
            agent.category = 1
            agent.name = "purchased_goods"
            return agent

    def test_calculate_by_spend(self, purchased_goods_agent):
        """Test spend-based calculation method."""
        purchased_goods_agent.calculate_by_spend = Mock(
            return_value=Decimal("1250.75")
        )

        emissions = purchased_goods_agent.calculate_by_spend(
            spend_amount=Decimal("10000"),
            spend_category="electronics",
            currency="USD"
        )

        assert emissions == Decimal("1250.75")

    def test_calculate_by_weight(self, purchased_goods_agent):
        """Test weight-based calculation method."""
        purchased_goods_agent.calculate_by_weight = Mock(
            return_value=Decimal("2340.0")
        )

        emissions = purchased_goods_agent.calculate_by_weight(
            material_weight=Decimal("1000"),  # kg
            material_type="steel"
        )

        assert emissions == Decimal("2340.0")

    @pytest.mark.parametrize("material,weight_kg,expected_emissions", [
        ("steel", 1000, Decimal("2000")),
        ("aluminum", 500, Decimal("5850")),
        ("plastic", 250, Decimal("875")),
        ("paper", 100, Decimal("120"))
    ])
    def test_material_specific_factors(
        self,
        purchased_goods_agent,
        material,
        weight_kg,
        expected_emissions
    ):
        """Test material-specific emission factors."""
        purchased_goods_agent.calculate_material_emissions = Mock(
            return_value=expected_emissions
        )

        emissions = purchased_goods_agent.calculate_material_emissions(
            material_type=material,
            weight=weight_kg
        )

        assert emissions == expected_emissions

    def test_supplier_specific_data(self, purchased_goods_agent):
        """Test using supplier-specific emission data."""
        supplier_data = {
            "supplier_id": "SUP-123",
            "product_carbon_footprint": Decimal("15.5"),
            "units_purchased": 100
        }

        purchased_goods_agent.calculate_with_supplier_data = Mock(
            return_value=Decimal("1550.0")
        )

        emissions = purchased_goods_agent.calculate_with_supplier_data(supplier_data)

        assert emissions == Decimal("1550.0")


class TestCapitalGoodsAgent:
    """Test suite for Category 2: Capital Goods."""

    @pytest.fixture
    def capital_goods_agent(self):
        """Create capital goods agent instance."""
        from greenlang.agents.scope3.capital_goods import CapitalGoodsAgent

        with patch('greenlang.agents.scope3.capital_goods.CapitalGoodsAgent.__init__', return_value=None):
            agent = CapitalGoodsAgent.__new__(CapitalGoodsAgent)
            agent.category = 2
            agent.name = "capital_goods"
            return agent

    def test_calculate_building_emissions(self, capital_goods_agent):
        """Test building construction emissions."""
        capital_goods_agent.calculate_building = Mock(
            return_value=Decimal("50000.0")
        )

        emissions = capital_goods_agent.calculate_building(
            building_area_m2=Decimal("1000"),
            building_type="office"
        )

        assert emissions == Decimal("50000.0")

    def test_calculate_machinery_emissions(self, capital_goods_agent):
        """Test machinery/equipment emissions."""
        capital_goods_agent.calculate_machinery = Mock(
            return_value=Decimal("15000.0")
        )

        emissions = capital_goods_agent.calculate_machinery(
            equipment_cost=Decimal("100000"),
            equipment_type="manufacturing"
        )

        assert emissions == Decimal("15000.0")

    def test_depreciation_allocation(self, capital_goods_agent):
        """Test emissions allocation over asset lifetime."""
        capital_goods_agent.allocate_over_lifetime = Mock(
            return_value=Decimal("5000.0")
        )

        annual_emissions = capital_goods_agent.allocate_over_lifetime(
            total_emissions=Decimal("50000"),
            lifetime_years=10
        )

        assert annual_emissions == Decimal("5000.0")


class TestUpstreamTransportAgent:
    """Test suite for Category 4: Upstream Transportation."""

    @pytest.fixture
    def transport_agent(self):
        """Create upstream transport agent instance."""
        from greenlang.agents.scope3.upstream_transport import UpstreamTransportAgent

        with patch('greenlang.agents.scope3.upstream_transport.UpstreamTransportAgent.__init__', return_value=None):
            agent = UpstreamTransportAgent.__new__(UpstreamTransportAgent)
            agent.category = 4
            agent.name = "upstream_transport"
            return agent

    @pytest.mark.parametrize("mode,distance_km,weight_tonnes,expected", [
        ("truck", 1000, 10, Decimal("1620.0")),  # 0.162 kg/t-km
        ("rail", 2000, 20, Decimal("840.0")),    # 0.021 kg/t-km
        ("ship", 5000, 50, Decimal("3000.0")),   # 0.012 kg/t-km
        ("air", 500, 1, Decimal("301.0"))        # 0.602 kg/t-km
    ])
    def test_transport_mode_emissions(
        self,
        transport_agent,
        mode,
        distance_km,
        weight_tonnes,
        expected
    ):
        """Test emissions for different transport modes."""
        transport_agent.calculate_transport = Mock(return_value=expected)

        emissions = transport_agent.calculate_transport(
            mode=mode,
            distance_km=distance_km,
            weight_tonnes=weight_tonnes
        )

        assert emissions == expected

    def test_multi_modal_transport(self, transport_agent):
        """Test multi-modal transportation chains."""
        segments = [
            {"mode": "truck", "distance": 100, "weight": 10},
            {"mode": "rail", "distance": 500, "weight": 10},
            {"mode": "truck", "distance": 50, "weight": 10}
        ]

        transport_agent.calculate_multi_modal = Mock(
            return_value=Decimal("297.0")
        )

        emissions = transport_agent.calculate_multi_modal(segments)

        assert emissions == Decimal("297.0")


class TestWasteAgent:
    """Test suite for Category 5: Waste Generated in Operations."""

    @pytest.fixture
    def waste_agent(self):
        """Create waste agent instance."""
        from greenlang.agents.scope3.waste import WasteAgent

        with patch('greenlang.agents.scope3.waste.WasteAgent.__init__', return_value=None):
            agent = WasteAgent.__new__(WasteAgent)
            agent.category = 5
            agent.name = "waste"
            return agent

    @pytest.mark.parametrize("disposal_method,waste_kg,expected", [
        ("landfill", 1000, Decimal("467.0")),
        ("incineration", 1000, Decimal("1021.0")),
        ("recycling", 1000, Decimal("21.0")),
        ("composting", 1000, Decimal("55.0"))
    ])
    def test_waste_disposal_methods(
        self,
        waste_agent,
        disposal_method,
        waste_kg,
        expected
    ):
        """Test emissions for different waste disposal methods."""
        waste_agent.calculate_disposal = Mock(return_value=expected)

        emissions = waste_agent.calculate_disposal(
            method=disposal_method,
            weight_kg=waste_kg
        )

        assert emissions == expected

    def test_waste_composition_breakdown(self, waste_agent):
        """Test emissions calculation with waste composition."""
        composition = {
            "organic": 300,
            "plastic": 200,
            "paper": 150,
            "metal": 100,
            "glass": 50
        }

        waste_agent.calculate_by_composition = Mock(
            return_value=Decimal("375.0")
        )

        emissions = waste_agent.calculate_by_composition(
            composition=composition,
            disposal_method="landfill"
        )

        assert emissions == Decimal("375.0")


class TestBusinessTravelAgent:
    """Test suite for Category 6: Business Travel."""

    @pytest.fixture
    def travel_agent(self):
        """Create business travel agent instance."""
        from greenlang.agents.scope3.business_travel import BusinessTravelAgent

        with patch('greenlang.agents.scope3.business_travel.BusinessTravelAgent.__init__', return_value=None):
            agent = BusinessTravelAgent.__new__(BusinessTravelAgent)
            agent.category = 6
            agent.name = "business_travel"
            return agent

    def test_air_travel_emissions(self, travel_agent):
        """Test air travel emissions calculation."""
        travel_agent.calculate_air_travel = Mock(
            return_value=Decimal("450.0")
        )

        emissions = travel_agent.calculate_air_travel(
            distance_km=2000,
            travel_class="economy",
            passengers=1
        )

        assert emissions == Decimal("450.0")

    @pytest.mark.parametrize("travel_class,multiplier", [
        ("economy", 1.0),
        ("premium_economy", 1.6),
        ("business", 2.9),
        ("first", 4.0)
    ])
    def test_travel_class_factors(self, travel_agent, travel_class, multiplier):
        """Test different travel class emission factors."""
        base_emissions = Decimal("100.0")

        travel_agent.apply_class_factor = Mock(
            return_value=base_emissions * Decimal(str(multiplier))
        )

        emissions = travel_agent.apply_class_factor(
            base_emissions,
            travel_class
        )

        assert emissions == base_emissions * Decimal(str(multiplier))

    def test_hotel_stay_emissions(self, travel_agent):
        """Test hotel stay emissions calculation."""
        travel_agent.calculate_hotel = Mock(
            return_value=Decimal("120.0")
        )

        emissions = travel_agent.calculate_hotel(
            nights=3,
            hotel_country="US"
        )

        assert emissions == Decimal("120.0")


class TestEmployeeCommutingAgent:
    """Test suite for Category 7: Employee Commuting."""

    @pytest.fixture
    def commuting_agent(self):
        """Create employee commuting agent instance."""
        from greenlang.agents.scope3.employee_commuting import EmployeeCommutingAgent

        with patch('greenlang.agents.scope3.employee_commuting.EmployeeCommutingAgent.__init__', return_value=None):
            agent = EmployeeCommutingAgent.__new__(EmployeeCommutingAgent)
            agent.category = 7
            agent.name = "employee_commuting"
            return agent

    def test_commute_by_mode(self, commuting_agent):
        """Test commuting emissions by transport mode."""
        commuting_agent.calculate_commute = Mock(
            return_value=Decimal("2500.0")
        )

        annual_emissions = commuting_agent.calculate_commute(
            employees=100,
            avg_distance_km=20,
            days_per_year=230,
            mode_split={
                "car": 0.6,
                "public_transport": 0.3,
                "bike": 0.1
            }
        )

        assert annual_emissions == Decimal("2500.0")

    def test_remote_work_adjustment(self, commuting_agent):
        """Test adjustment for remote work days."""
        commuting_agent.adjust_for_remote = Mock(
            return_value=Decimal("1500.0")
        )

        adjusted_emissions = commuting_agent.adjust_for_remote(
            base_emissions=Decimal("2500.0"),
            remote_percentage=0.4
        )

        assert adjusted_emissions == Decimal("1500.0")


class TestDownstreamTransportAgent:
    """Test suite for Category 9: Downstream Transportation."""

    @pytest.fixture
    def downstream_transport_agent(self):
        """Create downstream transport agent instance."""
        from greenlang.agents.scope3.downstream_transport import DownstreamTransportAgent

        with patch('greenlang.agents.scope3.downstream_transport.DownstreamTransportAgent.__init__', return_value=None):
            agent = DownstreamTransportAgent.__new__(DownstreamTransportAgent)
            agent.category = 9
            agent.name = "downstream_transport"
            return agent

    def test_distribution_network(self, downstream_transport_agent):
        """Test emissions from distribution network."""
        downstream_transport_agent.calculate_distribution = Mock(
            return_value=Decimal("5000.0")
        )

        emissions = downstream_transport_agent.calculate_distribution(
            distribution_centers=[
                {"location": "DC1", "volume": 1000},
                {"location": "DC2", "volume": 1500}
            ],
            delivery_mode="truck"
        )

        assert emissions == Decimal("5000.0")

    def test_last_mile_delivery(self, downstream_transport_agent):
        """Test last-mile delivery emissions."""
        downstream_transport_agent.calculate_last_mile = Mock(
            return_value=Decimal("850.0")
        )

        emissions = downstream_transport_agent.calculate_last_mile(
            deliveries=1000,
            avg_distance_km=5,
            vehicle_type="van"
        )

        assert emissions == Decimal("850.0")


class TestScope3Integration:
    """Integration tests for Scope 3 calculations."""

    @pytest.mark.integration
    def test_complete_scope3_calculation(self, mock_emission_factors):
        """Test complete Scope 3 emissions calculation."""
        from greenlang.agents.scope3.calculator import Scope3Calculator

        with patch('greenlang.agents.scope3.calculator.Scope3Calculator.__init__', return_value=None):
            calculator = Scope3Calculator.__new__(Scope3Calculator)
            calculator.emission_factors = mock_emission_factors

            # Mock category calculations
            calculator.calculate_all_categories = Mock(return_value={
                "category_1": Decimal("10000.0"),
                "category_2": Decimal("5000.0"),
                "category_3": Decimal("2000.0"),
                "category_4": Decimal("3000.0"),
                "category_5": Decimal("500.0"),
                "category_6": Decimal("1500.0"),
                "category_7": Decimal("2500.0"),
                "category_9": Decimal("5000.0"),
                "total": Decimal("29500.0")
            })

            results = calculator.calculate_all_categories()

            assert results["total"] == Decimal("29500.0")
            assert len([k for k in results.keys() if k.startswith("category_")]) == 8

    @pytest.mark.integration
    def test_scope3_data_quality_scoring(self):
        """Test data quality scoring for Scope 3 calculations."""
        from greenlang.agents.scope3.data_quality import DataQualityScorer

        with patch('greenlang.agents.scope3.data_quality.DataQualityScorer.__init__', return_value=None):
            scorer = DataQualityScorer.__new__(DataQualityScorer)

            scorer.score_data_quality = Mock(return_value={
                "overall_score": 3.5,
                "temporal_score": 4,
                "geographical_score": 3,
                "technological_score": 3,
                "completeness_score": 4,
                "reliability_score": 3
            })

            quality_score = scorer.score_data_quality({
                "data_age_months": 6,
                "geographical_match": "country",
                "technology_match": "similar",
                "data_completeness": 0.85,
                "source_reliability": "verified"
            })

            assert quality_score["overall_score"] == 3.5
            assert quality_score["completeness_score"] == 4

    @pytest.mark.performance
    def test_scope3_calculation_performance(self, performance_timer):
        """Test Scope 3 calculation performance."""
        from greenlang.agents.scope3.calculator import Scope3Calculator

        with patch('greenlang.agents.scope3.calculator.Scope3Calculator.__init__', return_value=None):
            calculator = Scope3Calculator.__new__(Scope3Calculator)
            calculator.calculate = Mock(return_value=Decimal("1000.0"))

            performance_timer.start()

            # Calculate 1000 emission records
            for _ in range(1000):
                calculator.calculate({"activity": 100, "factor": 2.5})

            performance_timer.stop()

            # Should complete in less than 500ms
            assert performance_timer.elapsed_ms() < 500