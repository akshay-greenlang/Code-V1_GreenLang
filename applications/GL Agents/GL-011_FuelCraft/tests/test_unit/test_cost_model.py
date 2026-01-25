# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft - Cost Model Unit Tests

Comprehensive unit tests for the CostModel class ensuring:
- Deterministic cost calculations (zero hallucination)
- Correct formula implementations
- Decimal precision in all arithmetic
- Provenance hash generation
- Edge case handling

Author: GL-TestEngineer
Date: 2025-01-01
Version: 1.0.0
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.cost_model import (
    CostModel,
    CostBreakdown,
    CostComponent,
    CostCategory,
    PurchaseCostParams,
    LogisticsCostParams,
    StorageCostParams,
    ContractPenaltyParams,
    CarbonCostParams,
    RiskCostParams,
    PricingType,
    LogisticsMode,
    CarbonScheme,
)


class TestCostModelInitialization:
    """Tests for CostModel initialization."""

    def test_default_initialization(self):
        """Test default CostModel initialization."""
        model = CostModel()
        assert model._currency == "USD"
        assert model._calculation_steps == []

    def test_custom_currency(self):
        """Test CostModel with custom currency."""
        model = CostModel(currency="EUR")
        assert model._currency == "EUR"


class TestPurchaseCostCalculation:
    """Tests for purchase cost calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_spot_pricing_basic(self, cost_model):
        """Test basic spot pricing calculation."""
        params = PurchaseCostParams(
            fuel_id="diesel",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.025")
        )

        cost, component = cost_model.calculate_purchase_cost(
            quantity_mj=Decimal("100000"),
            params=params
        )

        expected_cost = Decimal("100000") * Decimal("0.025")
        assert cost == expected_cost.quantize(Decimal("0.01"))
        assert component.category == CostCategory.PURCHASE
        assert component.fuel_id == "diesel" or "diesel" in component.name

    def test_contract_pricing(self, cost_model):
        """Test contract pricing calculation."""
        params = PurchaseCostParams(
            fuel_id="natural_gas",
            pricing_type=PricingType.CONTRACT,
            spot_price_per_mj=Decimal("0.030"),
            contract_price_per_mj=Decimal("0.025")
        )

        cost, component = cost_model.calculate_purchase_cost(
            quantity_mj=Decimal("50000"),
            params=params
        )

        # Contract price should be used
        expected_cost = Decimal("50000") * Decimal("0.025")
        assert cost == expected_cost.quantize(Decimal("0.01"))

    def test_index_adjustment(self, cost_model):
        """Test index adjustment on spot price."""
        params = PurchaseCostParams(
            fuel_id="diesel",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.020"),
            index_adjustment=Decimal("1.10")
        )

        cost, component = cost_model.calculate_purchase_cost(
            quantity_mj=Decimal("10000"),
            params=params
        )

        # Price = spot * index = 0.020 * 1.10 = 0.022
        expected_cost = Decimal("10000") * Decimal("0.022")
        assert cost == expected_cost.quantize(Decimal("0.01"))

    def test_volume_discount(self, cost_model):
        """Test volume discount application."""
        params = PurchaseCostParams(
            fuel_id="hfo",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.015"),
            volume_discount_threshold_mj=Decimal("50000"),
            volume_discount_pct=Decimal("5.0")
        )

        # Order above threshold
        cost, component = cost_model.calculate_purchase_cost(
            quantity_mj=Decimal("100000"),
            params=params
        )

        base_cost = Decimal("100000") * Decimal("0.015")
        discount = base_cost * Decimal("5.0") / Decimal("100")
        expected_cost = base_cost - discount
        assert cost == expected_cost.quantize(Decimal("0.01"))

    def test_volume_discount_below_threshold(self, cost_model):
        """Test no discount below threshold."""
        params = PurchaseCostParams(
            fuel_id="hfo",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.015"),
            volume_discount_threshold_mj=Decimal("50000"),
            volume_discount_pct=Decimal("5.0")
        )

        # Order below threshold - no discount
        cost, component = cost_model.calculate_purchase_cost(
            quantity_mj=Decimal("30000"),
            params=params
        )

        expected_cost = Decimal("30000") * Decimal("0.015")
        assert cost == expected_cost.quantize(Decimal("0.01"))

    def test_zero_quantity(self, cost_model):
        """Test zero quantity purchase."""
        params = PurchaseCostParams(
            fuel_id="diesel",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.025")
        )

        cost, component = cost_model.calculate_purchase_cost(
            quantity_mj=Decimal("0"),
            params=params
        )

        assert cost == Decimal("0.00")


class TestLogisticsCostCalculation:
    """Tests for logistics cost calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_basic_logistics_cost(self, cost_model):
        """Test basic logistics cost calculation."""
        params = LogisticsCostParams(
            mode=LogisticsMode.TRUCK,
            base_rate_per_mj=Decimal("0.001"),
            fixed_delivery_fee=Decimal("500")
        )

        cost, component = cost_model.calculate_logistics_cost(
            quantity_mj=Decimal("100000"),
            params=params,
            num_deliveries=1
        )

        expected = Decimal("100000") * Decimal("0.001") + Decimal("500")
        assert cost == expected.quantize(Decimal("0.01"))
        assert component.category == CostCategory.LOGISTICS

    def test_distance_based_cost(self, cost_model):
        """Test distance-based logistics cost."""
        params = LogisticsCostParams(
            mode=LogisticsMode.PIPELINE,
            base_rate_per_mj=Decimal("0.0005"),
            distance_km=Decimal("100"),
            distance_rate_per_mj_km=Decimal("0.00001")
        )

        cost, component = cost_model.calculate_logistics_cost(
            quantity_mj=Decimal("50000"),
            params=params
        )

        base = Decimal("50000") * Decimal("0.0005")
        distance = Decimal("50000") * Decimal("100") * Decimal("0.00001")
        expected = base + distance
        assert cost == expected.quantize(Decimal("0.01"))

    def test_multiple_deliveries(self, cost_model):
        """Test cost with multiple deliveries."""
        params = LogisticsCostParams(
            mode=LogisticsMode.TRUCK,
            base_rate_per_mj=Decimal("0.001"),
            fixed_delivery_fee=Decimal("200"),
            port_charges_per_delivery=Decimal("50")
        )

        cost, component = cost_model.calculate_logistics_cost(
            quantity_mj=Decimal("100000"),
            params=params,
            num_deliveries=5
        )

        transport = Decimal("100000") * Decimal("0.001")
        delivery_fees = Decimal("5") * Decimal("200")
        port_charges = Decimal("5") * Decimal("50")
        expected = transport + delivery_fees + port_charges
        assert cost == expected.quantize(Decimal("0.01"))


class TestStorageCostCalculation:
    """Tests for storage cost calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_basic_storage_cost(self, cost_model):
        """Test basic holding cost calculation."""
        params = StorageCostParams(
            tank_id="TANK-001",
            holding_cost_per_mj_day=Decimal("0.0001"),
            insurance_rate_annual_pct=Decimal("0"),
            capital_cost_per_mj_day=Decimal("0")
        )

        cost, component = cost_model.calculate_storage_cost(
            average_inventory_mj=Decimal("100000"),
            days=30,
            params=params
        )

        expected = Decimal("100000") * Decimal("30") * Decimal("0.0001")
        assert cost == expected.quantize(Decimal("0.01"))

    def test_storage_with_insurance(self, cost_model):
        """Test storage cost including insurance."""
        params = StorageCostParams(
            tank_id="TANK-001",
            holding_cost_per_mj_day=Decimal("0.0001"),
            insurance_rate_annual_pct=Decimal("0.365"),  # ~0.001% daily
            capital_cost_per_mj_day=Decimal("0")
        )

        cost, component = cost_model.calculate_storage_cost(
            average_inventory_mj=Decimal("100000"),
            days=365,
            params=params
        )

        holding = Decimal("100000") * Decimal("365") * Decimal("0.0001")
        insurance = Decimal("100000") * Decimal("365") * (Decimal("0.00365") / Decimal("365"))
        expected = holding + insurance
        assert abs(cost - expected.quantize(Decimal("0.01"))) < Decimal("1.00")


class TestLossCostCalculation:
    """Tests for loss cost calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_basic_loss_cost(self, cost_model):
        """Test basic inventory loss cost."""
        params = StorageCostParams(
            tank_id="TANK-001",
            holding_cost_per_mj_day=Decimal("0"),
            loss_rate_per_day_pct=Decimal("0.1")
        )

        cost, component = cost_model.calculate_loss_cost(
            inventory_mj=Decimal("100000"),
            days=10,
            params=params,
            fuel_price_per_mj=Decimal("0.025")
        )

        # Compound loss = 1 - (1 - 0.001)^10
        daily_rate = Decimal("0.001")
        retention = (Decimal("1") - daily_rate) ** 10
        loss_factor = Decimal("1") - retention
        loss_quantity = Decimal("100000") * loss_factor
        expected = loss_quantity * Decimal("0.025")
        assert abs(cost - expected.quantize(Decimal("0.01"))) < Decimal("1.00")


class TestPenaltyCostCalculation:
    """Tests for contract penalty calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_no_penalty_within_limits(self, cost_model):
        """Test no penalty when within contract limits."""
        params = ContractPenaltyParams(
            contract_id="CONTRACT-001",
            fuel_id="diesel",
            min_take_mj=Decimal("50000"),
            max_take_mj=Decimal("100000"),
            shortfall_penalty_per_mj=Decimal("0.010")
        )

        cost, component = cost_model.calculate_penalty_cost(
            actual_take_mj=Decimal("75000"),
            params=params
        )

        assert cost == Decimal("0.00")

    def test_shortfall_penalty(self, cost_model):
        """Test penalty for shortfall below minimum."""
        params = ContractPenaltyParams(
            contract_id="CONTRACT-001",
            fuel_id="diesel",
            min_take_mj=Decimal("50000"),
            max_take_mj=Decimal("100000"),
            shortfall_penalty_per_mj=Decimal("0.010")
        )

        cost, component = cost_model.calculate_penalty_cost(
            actual_take_mj=Decimal("30000"),
            params=params
        )

        shortfall = Decimal("50000") - Decimal("30000")
        expected = shortfall * Decimal("0.010")
        assert cost == expected.quantize(Decimal("0.01"))

    def test_excess_penalty(self, cost_model):
        """Test penalty for exceeding maximum."""
        params = ContractPenaltyParams(
            contract_id="CONTRACT-001",
            fuel_id="diesel",
            min_take_mj=Decimal("50000"),
            max_take_mj=Decimal("100000"),
            shortfall_penalty_per_mj=Decimal("0.010"),
            excess_penalty_per_mj=Decimal("0.005")
        )

        cost, component = cost_model.calculate_penalty_cost(
            actual_take_mj=Decimal("120000"),
            params=params
        )

        excess = Decimal("120000") - Decimal("100000")
        expected = excess * Decimal("0.005")
        assert cost == expected.quantize(Decimal("0.01"))


class TestCarbonCostCalculation:
    """Tests for carbon cost calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_basic_carbon_cost(self, cost_model):
        """Test basic carbon cost calculation."""
        params = CarbonCostParams(
            scheme=CarbonScheme.EU_ETS,
            carbon_price_per_kg_co2e=Decimal("0.080"),
            carbon_intensity_kg_co2e_mj=Decimal("0.0741")
        )

        cost, component, emissions = cost_model.calculate_carbon_cost(
            energy_mj=Decimal("100000"),
            params=params
        )

        expected_emissions = Decimal("100000") * Decimal("0.0741")
        expected_cost = expected_emissions * Decimal("0.080")
        assert cost == expected_cost.quantize(Decimal("0.01"))
        assert emissions == expected_emissions

    def test_carbon_cost_with_free_allowance(self, cost_model):
        """Test carbon cost with free emission allowance."""
        params = CarbonCostParams(
            scheme=CarbonScheme.EU_ETS,
            carbon_price_per_kg_co2e=Decimal("0.080"),
            carbon_intensity_kg_co2e_mj=Decimal("0.0741"),
            free_allowance_kg_co2e=Decimal("5000")
        )

        cost, component, emissions = cost_model.calculate_carbon_cost(
            energy_mj=Decimal("100000"),
            params=params
        )

        total_emissions = Decimal("100000") * Decimal("0.0741")
        net_emissions = total_emissions - Decimal("5000")
        expected_cost = net_emissions * Decimal("0.080")
        assert cost == expected_cost.quantize(Decimal("0.01"))

    def test_carbon_cost_with_compliance_factor(self, cost_model):
        """Test carbon cost with compliance factor."""
        params = CarbonCostParams(
            scheme=CarbonScheme.EU_ETS,
            carbon_price_per_kg_co2e=Decimal("0.080"),
            carbon_intensity_kg_co2e_mj=Decimal("0.0741"),
            compliance_factor=Decimal("0.5")
        )

        cost, component, emissions = cost_model.calculate_carbon_cost(
            energy_mj=Decimal("100000"),
            params=params
        )

        total_emissions = Decimal("100000") * Decimal("0.0741")
        taxable_emissions = total_emissions * Decimal("0.5")
        expected_cost = taxable_emissions * Decimal("0.080")
        assert cost == expected_cost.quantize(Decimal("0.01"))


class TestRiskCostCalculation:
    """Tests for risk cost calculations."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_basic_risk_premium(self, cost_model):
        """Test basic risk premium calculation."""
        params = RiskCostParams(
            volatility_premium_pct=Decimal("2.0"),
            supply_risk_premium_pct=Decimal("1.0")
        )

        cost, component = cost_model.calculate_risk_cost(
            base_cost=Decimal("100000"),
            params=params
        )

        expected = Decimal("100000") * Decimal("3.0") / Decimal("100")
        assert cost == expected.quantize(Decimal("0.01"))

    def test_all_risk_components(self, cost_model):
        """Test risk calculation with all components."""
        params = RiskCostParams(
            volatility_premium_pct=Decimal("2.0"),
            supply_risk_premium_pct=Decimal("1.5"),
            counterparty_risk_pct=Decimal("0.5"),
            hedging_cost_pct=Decimal("1.0")
        )

        cost, component = cost_model.calculate_risk_cost(
            base_cost=Decimal("50000"),
            params=params
        )

        total_pct = Decimal("2.0") + Decimal("1.5") + Decimal("0.5") + Decimal("1.0")
        expected = Decimal("50000") * total_pct / Decimal("100")
        assert cost == expected.quantize(Decimal("0.01"))


class TestTotalCostCalculation:
    """Tests for total cost breakdown calculation."""

    @pytest.fixture
    def cost_model(self):
        """Create CostModel instance for tests."""
        return CostModel()

    def test_total_cost_aggregation(self, cost_model):
        """Test aggregation of all cost components."""
        # Create sample components
        purchase = [(Decimal("1000.00"), CostComponent(
            category=CostCategory.PURCHASE,
            name="Purchase_diesel",
            amount=Decimal("1000.00"),
            unit="USD",
            quantity=Decimal("40000"),
            rate=Decimal("0.025"),
            description="Purchase",
            calculation_formula="qty * rate",
            input_values={}
        ))]

        logistics = [(Decimal("50.00"), CostComponent(
            category=CostCategory.LOGISTICS,
            name="Logistics",
            amount=Decimal("50.00"),
            unit="USD",
            quantity=Decimal("40000"),
            rate=Decimal("0.00125"),
            description="Logistics",
            calculation_formula="qty * rate",
            input_values={}
        ))]

        carbon = [(Decimal("237.12"), CostComponent(
            category=CostCategory.CARBON,
            name="Carbon",
            amount=Decimal("237.12"),
            unit="USD",
            quantity=Decimal("2964"),
            rate=Decimal("0.080"),
            description="Carbon",
            calculation_formula="emissions * price",
            input_values={}
        ), Decimal("2964"))]

        breakdown = cost_model.calculate_total_cost(
            purchase_components=purchase,
            logistics_components=logistics,
            storage_components=[],
            loss_components=[],
            penalty_components=[],
            carbon_components=carbon,
            risk_components=[],
            total_energy_mj=Decimal("40000"),
            period_count=1
        )

        assert breakdown.purchase_cost == Decimal("1000.00")
        assert breakdown.logistics_cost == Decimal("50.00")
        assert breakdown.carbon_cost == Decimal("237.12")
        assert breakdown.total_cost == Decimal("1287.12")

    def test_cost_breakdown_provenance(self, cost_model):
        """Test provenance hash generation."""
        breakdown = cost_model.calculate_total_cost(
            purchase_components=[],
            logistics_components=[],
            storage_components=[],
            loss_components=[],
            penalty_components=[],
            carbon_components=[],
            risk_components=[],
            total_energy_mj=Decimal("0"),
            period_count=0
        )

        assert breakdown.provenance_hash != ""
        assert len(breakdown.provenance_hash) == 64  # SHA-256 hex


class TestCostBreakdownSerialization:
    """Tests for CostBreakdown serialization."""

    def test_to_dict(self):
        """Test CostBreakdown to_dict method."""
        breakdown = CostBreakdown(
            purchase_cost=Decimal("1000.00"),
            logistics_cost=Decimal("50.00"),
            storage_cost=Decimal("10.00"),
            loss_cost=Decimal("5.00"),
            penalty_cost=Decimal("0.00"),
            carbon_cost=Decimal("200.00"),
            risk_cost=Decimal("25.00"),
            total_cost=Decimal("1290.00"),
            components=[],
            cost_by_fuel={"diesel": Decimal("1000.00")},
            cost_by_period={1: Decimal("1290.00")}
        )

        result = breakdown.to_dict()

        assert result["purchase_cost"] == "1000.00"
        assert result["total_cost"] == "1290.00"
        assert "diesel" in result["cost_by_fuel"]
        assert "provenance_hash" in result

    def test_get_cost_summary(self):
        """Test human-readable summary generation."""
        breakdown = CostBreakdown(
            purchase_cost=Decimal("1000.00"),
            logistics_cost=Decimal("50.00"),
            storage_cost=Decimal("10.00"),
            loss_cost=Decimal("5.00"),
            penalty_cost=Decimal("0.00"),
            carbon_cost=Decimal("200.00"),
            risk_cost=Decimal("25.00"),
            total_cost=Decimal("1290.00"),
            components=[],
            cost_by_fuel={},
            cost_by_period={},
            total_energy_mj=Decimal("40000")
        )

        summary = breakdown.get_cost_summary()

        assert "COST BREAKDOWN SUMMARY" in summary
        assert "$1,000.00" in summary
        assert "TOTAL COST" in summary


class TestDeterminism:
    """Tests for deterministic calculation behavior."""

    def test_repeated_calculation_same_result(self):
        """Test that repeated calculations produce identical results."""
        model = CostModel()
        params = PurchaseCostParams(
            fuel_id="diesel",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.025123456789")
        )

        results = []
        for _ in range(10):
            cost, _ = model.calculate_purchase_cost(
                quantity_mj=Decimal("123456.789"),
                params=params
            )
            results.append(cost)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_decimal_precision_preserved(self):
        """Test that Decimal precision is maintained throughout."""
        model = CostModel()
        params = PurchaseCostParams(
            fuel_id="diesel",
            pricing_type=PricingType.SPOT,
            spot_price_per_mj=Decimal("0.0001234567890123456789")
        )

        cost, component = model.calculate_purchase_cost(
            quantity_mj=Decimal("1000000000"),  # 1 billion MJ
            params=params
        )

        # Result should not have floating point errors
        expected = Decimal("1000000000") * Decimal("0.0001234567890123456789")
        assert cost == expected.quantize(Decimal("0.01"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
