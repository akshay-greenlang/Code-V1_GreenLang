"""
GL-011 FUELCRAFT - Cost Optimization Tests

Unit tests for CostOptimizer including total cost of ownership
calculations, multi-objective optimization, and carbon cost integration.
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_011_fuel_optimization.cost_optimization import (
    CostOptimizer,
    TotalCostInput,
    TotalCostOutput,
    CostBreakdown,
    CO2_EMISSION_FACTORS,
    MAINTENANCE_FACTORS,
    RELIABILITY_FACTORS,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    CostOptimizationConfig,
    OptimizationMode,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelPrice,
    CostAnalysis,
)


class TestCO2EmissionFactors:
    """Tests for CO2 emission factor constants."""

    def test_natural_gas_emission_factor(self):
        """Test natural gas emission factor."""
        assert CO2_EMISSION_FACTORS["natural_gas"] == pytest.approx(53.06, rel=0.01)

    def test_fuel_oil_emission_factor(self):
        """Test fuel oil emission factor."""
        assert CO2_EMISSION_FACTORS["no2_fuel_oil"] == pytest.approx(73.16, rel=0.01)
        assert CO2_EMISSION_FACTORS["no6_fuel_oil"] == pytest.approx(75.10, rel=0.01)

    def test_coal_emission_factor(self):
        """Test coal emission factor."""
        assert CO2_EMISSION_FACTORS["coal_bituminous"] == pytest.approx(93.28, rel=0.01)

    def test_zero_carbon_fuels(self):
        """Test zero-carbon fuels have zero emission factor."""
        assert CO2_EMISSION_FACTORS["biomass_wood"] == 0.0
        assert CO2_EMISSION_FACTORS["biogas"] == 0.0
        assert CO2_EMISSION_FACTORS["hydrogen"] == 0.0
        assert CO2_EMISSION_FACTORS["rng"] == 0.0


class TestMaintenanceFactors:
    """Tests for maintenance factor constants."""

    def test_natural_gas_maintenance(self):
        """Test natural gas has baseline maintenance factor."""
        assert MAINTENANCE_FACTORS["natural_gas"] == 1.0

    def test_fuel_oil_higher_maintenance(self):
        """Test fuel oil has higher maintenance factor."""
        assert MAINTENANCE_FACTORS["no2_fuel_oil"] > MAINTENANCE_FACTORS["natural_gas"]
        assert MAINTENANCE_FACTORS["no6_fuel_oil"] > MAINTENANCE_FACTORS["no2_fuel_oil"]

    def test_coal_highest_maintenance(self):
        """Test coal has highest maintenance factor."""
        assert MAINTENANCE_FACTORS["coal_bituminous"] == max(MAINTENANCE_FACTORS.values())


class TestReliabilityFactors:
    """Tests for reliability factor constants."""

    def test_natural_gas_high_reliability(self):
        """Test natural gas has high reliability."""
        assert RELIABILITY_FACTORS["natural_gas"] == 0.995

    def test_all_factors_between_0_and_1(self):
        """Test all reliability factors are valid probabilities."""
        for fuel, factor in RELIABILITY_FACTORS.items():
            assert 0.0 <= factor <= 1.0, f"{fuel} has invalid reliability factor"


class TestCostOptimizer:
    """Tests for CostOptimizer class."""

    def test_optimizer_initialization(self, cost_optimizer):
        """Test optimizer initialization."""
        assert cost_optimizer.config is not None
        assert cost_optimizer.optimization_count == 0

    def test_optimization_count_increments(self, cost_optimizer, total_cost_input):
        """Test optimization count increments."""
        initial = cost_optimizer.optimization_count

        cost_optimizer.optimize(total_cost_input)

        assert cost_optimizer.optimization_count == initial + 1


class TestCostBreakdownCalculation:
    """Tests for cost breakdown calculation."""

    def test_fuel_purchase_cost(self, cost_optimizer, total_cost_input):
        """Test fuel purchase cost calculation."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        # Annual fuel consumption = 50 MMBTU/hr * 8000 hrs / 0.82 efficiency
        # Fuel cost = consumption * price
        expected_consumption = 50.0 * 8000.0 / 0.82
        expected_cost = expected_consumption * total_cost_input.fuel_prices["natural_gas"].commodity_price

        assert breakdown.fuel_purchase_cost == pytest.approx(expected_cost, rel=0.01)

    def test_transport_cost(self, cost_optimizer, total_cost_input):
        """Test transport cost calculation."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        assert breakdown.transport_cost >= 0

    def test_storage_cost(self, cost_optimizer, total_cost_input):
        """Test storage cost calculation."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        # Storage cost = 2% of fuel purchase
        expected = breakdown.fuel_purchase_cost * 0.02
        assert breakdown.storage_cost == pytest.approx(expected, rel=0.01)

    def test_carbon_cost(self, cost_optimizer, total_cost_input):
        """Test carbon cost calculation."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        # Carbon cost = CO2 tons * carbon price
        expected_co2_tons = breakdown.annual_co2_tons
        expected_cost = expected_co2_tons * total_cost_input.carbon_price_usd_ton

        assert breakdown.carbon_cost == pytest.approx(expected_cost, rel=0.01)

    def test_maintenance_cost(self, cost_optimizer, total_cost_input):
        """Test maintenance cost calculation."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        # Maintenance = base * factor
        expected = total_cost_input.base_maintenance_cost_usd_year * MAINTENANCE_FACTORS["natural_gas"]
        assert breakdown.maintenance_cost == pytest.approx(expected, rel=0.01)

    def test_total_cost_sum(self, cost_optimizer, total_cost_input):
        """Test total cost equals sum of components."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        expected_total = (
            breakdown.fuel_purchase_cost +
            breakdown.transport_cost +
            breakdown.storage_cost +
            breakdown.carbon_cost +
            breakdown.maintenance_cost +
            breakdown.efficiency_adjustment
        )

        assert breakdown.total_annual_cost == pytest.approx(expected_total, rel=0.01)


class TestEfficiencyImpact:
    """Tests for efficiency impact on cost."""

    def test_lower_efficiency_increases_cost(self, cost_optimization_config, all_fuel_prices, all_fuel_properties):
        """Test lower efficiency increases fuel cost."""
        optimizer = CostOptimizer(cost_optimization_config)

        # Test with different efficiencies
        input_high_eff = TotalCostInput(
            fuel_options=["natural_gas"],
            fuel_prices=all_fuel_prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
            equipment_efficiency={"natural_gas": 90.0},
        )

        input_low_eff = TotalCostInput(
            fuel_options=["natural_gas"],
            fuel_prices=all_fuel_prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
            equipment_efficiency={"natural_gas": 75.0},
        )

        result_high = optimizer.optimize(input_high_eff)
        result_low = optimizer.optimize(input_low_eff)

        # Lower efficiency = higher cost
        assert result_low.total_cost_usd > result_high.total_cost_usd


class TestOptimizationModes:
    """Tests for different optimization modes."""

    def test_minimum_cost_mode(self, all_fuel_prices, all_fuel_properties):
        """Test minimum cost optimization mode."""
        config = CostOptimizationConfig(mode=OptimizationMode.MINIMUM_COST)
        optimizer = CostOptimizer(config)

        input_data = TotalCostInput(
            fuel_options=["natural_gas", "no2_fuel_oil", "lpg_propane"],
            fuel_prices=all_fuel_prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
        )

        result = optimizer.optimize(input_data)

        # Should select lowest cost fuel
        assert result.optimal_fuel == "natural_gas"  # Cheapest in fixture
        assert result.optimization_mode == OptimizationMode.MINIMUM_COST

    def test_minimum_emissions_mode(self, all_fuel_prices, all_fuel_properties):
        """Test minimum emissions optimization mode."""
        config = CostOptimizationConfig(mode=OptimizationMode.MINIMUM_EMISSIONS)
        optimizer = CostOptimizer(config)

        # Add hydrogen as zero-carbon option
        prices = dict(all_fuel_prices)
        prices["hydrogen"] = FuelPrice(
            fuel_type="hydrogen",
            price=15.00,
            commodity_price=14.00,
            source="test",
        )

        input_data = TotalCostInput(
            fuel_options=["natural_gas", "hydrogen"],
            fuel_prices=prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
        )

        result = optimizer.optimize(input_data)

        # Should select lowest emissions fuel
        assert result.optimal_fuel == "hydrogen"
        assert result.optimization_mode == OptimizationMode.MINIMUM_EMISSIONS

    def test_reliability_mode(self, all_fuel_prices):
        """Test reliability optimization mode."""
        config = CostOptimizationConfig(mode=OptimizationMode.RELIABILITY)
        optimizer = CostOptimizer(config)

        input_data = TotalCostInput(
            fuel_options=["natural_gas", "no2_fuel_oil", "biogas"],
            fuel_prices=all_fuel_prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
        )

        result = optimizer.optimize(input_data)

        # Should select most reliable fuel (natural gas = 0.995)
        assert result.optimal_fuel == "natural_gas"

    def test_balanced_mode(self, all_fuel_prices):
        """Test balanced optimization mode."""
        config = CostOptimizationConfig(
            mode=OptimizationMode.BALANCED,
            cost_weight=0.4,
            emissions_weight=0.4,
            reliability_weight=0.2,
        )
        optimizer = CostOptimizer(config)

        input_data = TotalCostInput(
            fuel_options=["natural_gas", "no2_fuel_oil"],
            fuel_prices=all_fuel_prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
        )

        result = optimizer.optimize(input_data)

        assert result.optimization_mode == OptimizationMode.BALANCED
        # Weighted score should be calculated
        assert 0.0 <= result.weighted_score <= 1.0


class TestMultiObjectiveScoring:
    """Tests for multi-objective scoring."""

    def test_cost_score_calculation(self, cost_optimizer, total_cost_input):
        """Test cost score calculation."""
        result = cost_optimizer.optimize(total_cost_input)

        # Cost score should be between 0 and 1
        assert 0.0 <= result.cost_score <= 1.0

    def test_emissions_score_calculation(self, cost_optimizer, total_cost_input):
        """Test emissions score calculation."""
        result = cost_optimizer.optimize(total_cost_input)

        assert 0.0 <= result.emissions_score <= 1.0

    def test_reliability_score_calculation(self, cost_optimizer, total_cost_input):
        """Test reliability score calculation."""
        result = cost_optimizer.optimize(total_cost_input)

        assert 0.0 <= result.reliability_score <= 1.0

    def test_weighted_score_calculation(self, cost_optimization_config, total_cost_input):
        """Test weighted score calculation."""
        config = CostOptimizationConfig(
            cost_weight=0.5,
            emissions_weight=0.3,
            reliability_weight=0.2,
        )
        optimizer = CostOptimizer(config)

        result = optimizer.optimize(total_cost_input)

        expected_weighted = (
            config.cost_weight * result.cost_score +
            config.emissions_weight * result.emissions_score +
            config.reliability_weight * result.reliability_score
        )

        assert result.weighted_score == pytest.approx(expected_weighted, rel=0.01)


class TestFuelRankings:
    """Tests for fuel ranking functionality."""

    def test_fuel_rankings_sorted(self, cost_optimizer, total_cost_input):
        """Test fuel rankings are sorted by cost."""
        result = cost_optimizer.optimize(total_cost_input)

        # Rankings should be sorted by cost (ascending)
        costs = [cost for fuel, cost in result.fuel_rankings]
        assert costs == sorted(costs)

    def test_all_fuels_ranked(self, cost_optimizer, total_cost_input):
        """Test all input fuels are ranked."""
        result = cost_optimizer.optimize(total_cost_input)

        ranked_fuels = {fuel for fuel, cost in result.fuel_rankings}
        input_fuels = set(total_cost_input.fuel_options)

        assert ranked_fuels == input_fuels


class TestSavingsCalculation:
    """Tests for savings calculation."""

    def test_savings_vs_current(self, all_fuel_prices):
        """Test savings calculation vs current fuel."""
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(config)

        # Create scenario where switch saves money
        prices = dict(all_fuel_prices)
        prices["no2_fuel_oil"] = FuelPrice(
            fuel_type="no2_fuel_oil",
            price=2.00,  # Much cheaper than current
            commodity_price=1.50,
            source="test",
        )

        input_data = TotalCostInput(
            fuel_options=["natural_gas", "no2_fuel_oil"],
            fuel_prices=prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
        )

        result = optimizer.optimize(input_data)

        # Should have positive savings
        assert result.savings_vs_current_usd > 0
        assert result.savings_vs_current_pct > 0

    def test_no_savings_when_current_is_optimal(self, cost_optimizer, total_cost_input):
        """Test no savings when current fuel is optimal."""
        result = cost_optimizer.optimize(total_cost_input)

        if result.optimal_fuel == total_cost_input.current_fuel:
            assert result.savings_vs_current_usd == 0.0


class TestCarbonPriceScenarios:
    """Tests for carbon price scenario analysis."""

    def test_compare_scenarios(self, cost_optimizer, total_cost_input):
        """Test comparing multiple carbon price scenarios."""
        carbon_prices = [25.0, 50.0, 100.0, 200.0]

        results = cost_optimizer.compare_scenarios(total_cost_input, carbon_prices)

        assert len(results) == 4
        assert 25.0 in results
        assert 200.0 in results

    def test_higher_carbon_price_higher_cost(self, cost_optimizer, total_cost_input):
        """Test higher carbon price increases fossil fuel cost."""
        results = cost_optimizer.compare_scenarios(
            total_cost_input,
            [25.0, 100.0],
        )

        # Higher carbon price = higher cost for fossil fuels
        assert results[100.0].carbon_cost_usd > results[25.0].carbon_cost_usd

    def test_carbon_price_changes_optimal(self, all_fuel_prices):
        """Test high carbon price can change optimal fuel."""
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(config)

        # Add hydrogen option
        prices = dict(all_fuel_prices)
        prices["hydrogen"] = FuelPrice(
            fuel_type="hydrogen",
            price=8.00,
            commodity_price=7.00,
            source="test",
        )

        input_low_carbon = TotalCostInput(
            fuel_options=["natural_gas", "hydrogen"],
            fuel_prices=prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
            carbon_price_usd_ton=10.0,  # Low carbon price
        )

        input_high_carbon = TotalCostInput(
            fuel_options=["natural_gas", "hydrogen"],
            fuel_prices=prices,
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
            carbon_price_usd_ton=300.0,  # High carbon price
        )

        result_low = optimizer.optimize(input_low_carbon)
        result_high = optimizer.optimize(input_high_carbon)

        # With very high carbon price, hydrogen should become more competitive
        # (though may still not be optimal depending on exact prices)
        assert result_high.carbon_cost_usd <= result_low.carbon_cost_usd * 0.5 or \
               result_high.optimal_fuel == "hydrogen"


class TestCostPerUnit:
    """Tests for cost per unit calculations."""

    def test_cost_per_mmbtu(self, cost_optimizer, total_cost_input):
        """Test cost per MMBTU calculation."""
        result = cost_optimizer.optimize(total_cost_input)

        annual_heat = total_cost_input.heat_demand_mmbtu_hr * total_cost_input.operating_hours_year
        expected = result.total_cost_usd / annual_heat

        assert result.cost_per_mmbtu == pytest.approx(expected, rel=0.01)

    def test_cost_per_hour(self, cost_optimizer, total_cost_input):
        """Test cost per operating hour calculation."""
        result = cost_optimizer.optimize(total_cost_input)

        expected = result.total_cost_usd / total_cost_input.operating_hours_year

        assert result.cost_per_hour == pytest.approx(expected, rel=0.01)


class TestEmissionsOutput:
    """Tests for emissions output."""

    def test_annual_co2_tons(self, cost_optimizer, total_cost_input):
        """Test annual CO2 calculation."""
        result = cost_optimizer.optimize(total_cost_input)

        assert result.annual_co2_tons > 0

    def test_co2_intensity(self, cost_optimizer, total_cost_input):
        """Test CO2 intensity in output."""
        result = cost_optimizer.optimize(total_cost_input)

        # Should match emission factor for optimal fuel
        expected = CO2_EMISSION_FACTORS.get(result.optimal_fuel, 53.0)
        assert result.co2_intensity_kg_mmbtu == pytest.approx(expected, rel=0.01)


class TestCostAnalysisCreation:
    """Tests for CostAnalysis creation."""

    def test_create_cost_analysis(self, cost_optimizer, total_cost_input):
        """Test creating CostAnalysis from output."""
        output = cost_optimizer.optimize(total_cost_input)

        analysis = cost_optimizer.create_cost_analysis(output, total_cost_input)

        assert isinstance(analysis, CostAnalysis)
        assert analysis.period_hours == total_cost_input.operating_hours_year
        assert analysis.fuel_cost_usd == output.fuel_cost_usd
        assert analysis.total_cost_usd == output.total_cost_usd


class TestProvenanceTracking:
    """Tests for provenance tracking."""

    def test_provenance_hash_generated(self, cost_optimizer, total_cost_input):
        """Test provenance hash is generated."""
        result = cost_optimizer.optimize(total_cost_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(self, cost_optimizer, total_cost_input):
        """Test same input produces same hash."""
        result1 = cost_optimizer.optimize(total_cost_input)
        result2 = cost_optimizer.optimize(total_cost_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_analysis_timestamp(self, cost_optimizer, total_cost_input):
        """Test analysis timestamp is set."""
        result = cost_optimizer.optimize(total_cost_input)

        assert result.analysis_timestamp is not None
        assert result.analysis_timestamp.tzinfo is not None


class TestMissingFuelPrice:
    """Tests for handling missing fuel prices."""

    def test_missing_price_raises_error(self, cost_optimizer):
        """Test missing fuel price raises error."""
        input_data = TotalCostInput(
            fuel_options=["natural_gas", "unknown_fuel"],
            fuel_prices={"natural_gas": FuelPrice(
                fuel_type="natural_gas",
                price=3.50,
                commodity_price=3.00,
                source="test",
            )},
            heat_demand_mmbtu_hr=50.0,
            current_fuel="natural_gas",
        )

        with pytest.raises(ValueError, match="No price"):
            cost_optimizer.optimize(input_data)


class TestAvailabilityAdjustedCost:
    """Tests for availability-adjusted cost."""

    def test_availability_adjusted_cost(self, cost_optimizer, total_cost_input):
        """Test availability-adjusted cost calculation."""
        breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)

        # Adjusted cost = total / reliability
        expected = breakdown.total_annual_cost / RELIABILITY_FACTORS["natural_gas"]

        assert breakdown.availability_adjusted_cost == pytest.approx(expected, rel=0.01)

    def test_lower_reliability_higher_adjusted_cost(self, cost_optimizer, total_cost_input):
        """Test lower reliability increases adjusted cost."""
        ng_breakdown = cost_optimizer.analyze_fuel("natural_gas", total_cost_input)
        biogas_breakdown = cost_optimizer.analyze_fuel("biogas", total_cost_input)

        # Biogas has lower reliability, so adjusted cost increase should be larger
        ng_increase = ng_breakdown.availability_adjusted_cost - ng_breakdown.total_annual_cost
        biogas_increase = biogas_breakdown.availability_adjusted_cost - biogas_breakdown.total_annual_cost

        # Percentage increase should be larger for biogas
        ng_pct = ng_increase / ng_breakdown.total_annual_cost
        biogas_pct = biogas_increase / biogas_breakdown.total_annual_cost

        assert biogas_pct > ng_pct
