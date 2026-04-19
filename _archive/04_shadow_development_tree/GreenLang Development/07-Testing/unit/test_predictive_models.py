# -*- coding: utf-8 -*-
"""
Unit Tests for Predictive Models Module

Tests for TASK-106, TASK-109, and TASK-110 implementations:
- Fuel Price Prediction (fuel_price.py)
- Spare Parts Optimization (spare_parts.py)
- Production Impact Modeling (production_impact.py)

Test coverage targets:
- All public methods
- Edge cases
- Provenance tracking
- Deterministic behavior
- Input validation

Author: GreenLang Engineering Team
"""

import pytest
import math
from datetime import datetime, timedelta
from typing import List, Tuple

# Import modules under test
from greenlang.ml.predictive.fuel_price import (
    FuelPricePredictor,
    FuelType,
    ForecastHorizon,
    ModelType,
    VolatilityRegime,
    FuelPriceDataPoint,
    FuelPriceInput,
    FuelPriceOutput,
    forecast_natural_gas_price,
)

from greenlang.ml.predictive.spare_parts import (
    SparePartsOptimizer,
    PartCriticality,
    EquipmentType,
    StockingStrategy,
    DemandPattern,
    PartUsageRecord,
    SparePartInput,
    EOQResult,
    SafetyStockResult,
    calculate_eoq,
    calculate_safety_stock,
    recommend_stocking_strategy,
)

from greenlang.ml.predictive.production_impact import (
    ProductionImpactModeler,
    FailureMode,
    ImpactSeverity,
    EquipmentProfile,
    ProductionParameters,
    FailureScenario,
    ProductionImpactInput,
    estimate_downtime_cost,
    calculate_availability,
    calculate_risk_priority_number,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_fuel_prices() -> List[FuelPriceDataPoint]:
    """Generate sample fuel price data."""
    base_date = datetime(2024, 1, 1)
    prices = []
    for i in range(60):
        # Simulate price with trend and noise
        price = 3.0 + i * 0.01 + (i % 7) * 0.05
        prices.append(FuelPriceDataPoint(
            date=base_date + timedelta(days=i),
            price=price,
            unit="MMBtu"
        ))
    return prices


@pytest.fixture
def sample_spare_part_input() -> SparePartInput:
    """Create sample spare part input."""
    return SparePartInput(
        part_id="PUMP-SEAL-001",
        part_name="Mechanical Seal for Boiler Feed Pump",
        equipment_type=EquipmentType.PUMP,
        criticality=PartCriticality.ESSENTIAL,
        unit_cost=750.0,
        ordering_cost=100.0,
        holding_cost_rate=0.25,
        current_stock=3,
        lead_time_days=21,
        lead_time_std=5,
        annual_usage=12
    )


@pytest.fixture
def sample_equipment_profile() -> EquipmentProfile:
    """Create sample equipment profile."""
    return EquipmentProfile(
        equipment_id="BFP-001",
        equipment_name="Boiler Feed Pump",
        equipment_type="pump",
        rated_capacity=500,
        capacity_units="gpm",
        current_efficiency=0.82,
        mtbf_hours=6000,
        mttr_hours=6,
        hourly_operating_cost=50,
        hourly_labor_cost=85,
        energy_consumption_kwh=75
    )


@pytest.fixture
def sample_production_params() -> ProductionParameters:
    """Create sample production parameters."""
    return ProductionParameters(
        production_value_per_unit=150.0,
        hourly_production_rate=25.0,
        production_unit="ton",
        operating_hours_per_day=24,
        operating_days_per_year=350,
        current_utilization=0.85,
        quality_yield=0.97,
        rework_cost_per_unit=20,
        scrap_cost_per_unit=50,
        energy_cost_per_kwh=0.09,
        natural_gas_cost_per_mmbtu=4.50,
        customer_penalty_per_day=5000
    )


@pytest.fixture
def sample_failure_scenario() -> FailureScenario:
    """Create sample failure scenario."""
    return FailureScenario(
        scenario_id="FS-001",
        scenario_name="Pump Mechanical Seal Failure",
        failure_mode=FailureMode.MECHANICAL,
        probability=0.15,
        duration_hours_min=4,
        duration_hours_max=12,
        duration_hours_mode=6,
        capacity_loss_pct=100,
        quality_impact_pct=5,
        repair_cost_min=2000,
        repair_cost_max=6000,
        affected_equipment=["BFP-001"]
    )


# ============================================================================
# TASK-106: Fuel Price Prediction Tests
# ============================================================================

class TestFuelPricePredictor:
    """Tests for FuelPricePredictor class."""

    def test_init_default_precision(self):
        """Test initialization with default precision."""
        predictor = FuelPricePredictor()
        assert predictor.precision == 4

    def test_init_custom_precision(self):
        """Test initialization with custom precision."""
        predictor = FuelPricePredictor(precision=2)
        assert predictor.precision == 2

    def test_calculate_returns(self):
        """Test log return calculation."""
        predictor = FuelPricePredictor()
        prices = [100, 105, 103, 110]
        returns = predictor._calculate_returns(prices)

        assert len(returns) == 3
        assert abs(returns[0] - math.log(105/100)) < 0.0001
        assert abs(returns[1] - math.log(103/105)) < 0.0001
        assert abs(returns[2] - math.log(110/103)) < 0.0001

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        predictor = FuelPricePredictor()

        # Stable prices - low volatility
        stable_prices = [100 + i * 0.1 for i in range(60)]
        vol_stable = predictor._calculate_volatility(stable_prices)
        assert vol_stable.historical_volatility < 0.1

        # Volatile prices
        volatile_prices = [100 + (i % 10) * 5 for i in range(60)]
        vol_high = predictor._calculate_volatility(volatile_prices)
        assert vol_high.historical_volatility > vol_stable.historical_volatility

    def test_volatility_regime_classification(self):
        """Test volatility regime is correctly classified."""
        predictor = FuelPricePredictor()

        # Low volatility
        stable = [100 + i * 0.01 for i in range(60)]
        result = predictor._calculate_volatility(stable)
        assert result.volatility_regime in [VolatilityRegime.LOW, VolatilityRegime.NORMAL]

    def test_arima_forecast_length(self, sample_fuel_prices):
        """Test ARIMA forecast generates correct number of points."""
        predictor = FuelPricePredictor()
        prices = [p.price for p in sample_fuel_prices]
        dates = [p.date for p in sample_fuel_prices]

        forecasts = predictor._arima_forecast(prices, dates, 30, 0.95)

        assert len(forecasts) == 30
        assert all(f.predicted_price > 0 for f in forecasts)

    def test_arima_forecast_confidence_intervals(self, sample_fuel_prices):
        """Test ARIMA forecasts have valid confidence intervals."""
        predictor = FuelPricePredictor()
        prices = [p.price for p in sample_fuel_prices]
        dates = [p.date for p in sample_fuel_prices]

        forecasts = predictor._arima_forecast(prices, dates, 30, 0.95)

        for f in forecasts:
            assert f.lower_bound <= f.predicted_price <= f.upper_bound
            assert f.confidence_level == 0.95

    def test_exponential_smoothing_forecast(self, sample_fuel_prices):
        """Test exponential smoothing forecast."""
        predictor = FuelPricePredictor()
        prices = [p.price for p in sample_fuel_prices]
        dates = [p.date for p in sample_fuel_prices]

        forecasts = predictor._exponential_smoothing_forecast(
            prices, dates, 30, 0.95
        )

        assert len(forecasts) == 30
        assert all(f.predicted_price > 0 for f in forecasts)

    def test_seasonal_adjustment(self):
        """Test seasonal adjustment is applied."""
        predictor = FuelPricePredictor()

        # Winter forecasts should be higher for natural gas
        from greenlang.ml.predictive.fuel_price import ForecastPoint

        winter_date = datetime(2024, 1, 15)  # January
        summer_date = datetime(2024, 7, 15)  # July

        winter_forecast = ForecastPoint(
            date=winter_date,
            predicted_price=3.0,
            lower_bound=2.5,
            upper_bound=3.5,
            confidence_level=0.95
        )

        summer_forecast = ForecastPoint(
            date=summer_date,
            predicted_price=3.0,
            lower_bound=2.5,
            upper_bound=3.5,
            confidence_level=0.95
        )

        adjusted = predictor._apply_seasonal_adjustment(
            [winter_forecast, summer_forecast],
            FuelType.NATURAL_GAS
        )

        # Winter should be higher than summer for natural gas
        assert adjusted[0].predicted_price > adjusted[1].predicted_price

    def test_full_prediction_pipeline(self, sample_fuel_prices):
        """Test full prediction pipeline."""
        predictor = FuelPricePredictor()

        input_data = FuelPriceInput(
            fuel_type=FuelType.NATURAL_GAS,
            historical_prices=sample_fuel_prices,
            forecast_horizon=ForecastHorizon.DAYS_30,
            model_type=ModelType.ARIMA,
            confidence_level=0.95
        )

        result = predictor.predict(input_data)

        assert isinstance(result, FuelPriceOutput)
        assert result.fuel_type == FuelType.NATURAL_GAS
        assert len(result.forecasts) == 30
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_ensemble_model(self, sample_fuel_prices):
        """Test ensemble model combines ARIMA and ES."""
        predictor = FuelPricePredictor()

        input_data = FuelPriceInput(
            fuel_type=FuelType.NATURAL_GAS,
            historical_prices=sample_fuel_prices,
            forecast_horizon=ForecastHorizon.DAYS_30,
            model_type=ModelType.ENSEMBLE,
            confidence_level=0.95
        )

        result = predictor.predict(input_data)
        assert result.model_type == ModelType.ENSEMBLE
        assert len(result.forecasts) == 30

    def test_market_factor_analysis(self, sample_fuel_prices):
        """Test market factor analysis is generated."""
        predictor = FuelPricePredictor()

        input_data = FuelPriceInput(
            fuel_type=FuelType.NATURAL_GAS,
            historical_prices=sample_fuel_prices,
            forecast_horizon=ForecastHorizon.DAYS_30,
            include_market_factors=True
        )

        result = predictor.predict(input_data)
        assert len(result.market_factor_analysis) > 0

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        predictor = FuelPricePredictor()

        inputs = {"fuel": "natural_gas", "n": 60}
        outputs = {"avg_price": 3.5}

        hash1 = predictor._calculate_provenance(inputs, outputs)
        hash2 = predictor._calculate_provenance(inputs, outputs)

        assert hash1 == hash2

    def test_convenience_function(self):
        """Test convenience function for natural gas forecast."""
        base_date = datetime(2024, 1, 1)
        prices = [
            (base_date + timedelta(days=i), 3.0 + i * 0.01)
            for i in range(60)
        ]

        result = forecast_natural_gas_price(prices)
        assert result.fuel_type == FuelType.NATURAL_GAS
        assert result.forecast_30d is not None


# ============================================================================
# TASK-109: Spare Parts Optimization Tests
# ============================================================================

class TestSparePartsOptimizer:
    """Tests for SparePartsOptimizer class."""

    def test_init_default_precision(self):
        """Test initialization with default precision."""
        optimizer = SparePartsOptimizer()
        assert optimizer.precision == 4

    def test_eoq_calculation_basic(self):
        """Test basic EOQ calculation."""
        optimizer = SparePartsOptimizer()

        # Known case: D=1000, S=50, H=5 (unit_cost=100, rate=0.05)
        # EOQ = sqrt(2*1000*50/5) = sqrt(20000) = 141.42
        result = optimizer._calculate_eoq(
            annual_demand=1000,
            unit_cost=100,
            ordering_cost=50,
            holding_cost_rate=0.05
        )

        assert result.eoq == 141 or result.eoq == 142  # Rounding
        assert result.orders_per_year > 0
        assert result.total_annual_cost > 0

    def test_eoq_zero_demand(self):
        """Test EOQ with zero demand returns minimum."""
        optimizer = SparePartsOptimizer()

        result = optimizer._calculate_eoq(
            annual_demand=0,
            unit_cost=100,
            ordering_cost=50,
            holding_cost_rate=0.25
        )

        assert result.eoq == 1  # Minimum

    def test_safety_stock_calculation(self):
        """Test safety stock calculation."""
        optimizer = SparePartsOptimizer()

        result = optimizer._calculate_safety_stock(
            daily_demand=1.0,
            demand_std=0.3,
            lead_time_days=14,
            lead_time_std=2,
            service_level=0.95
        )

        assert result.safety_stock >= 0
        assert result.reorder_point > result.safety_stock
        assert result.service_level == 0.95
        assert result.stockout_risk == pytest.approx(0.05, rel=0.01)

    def test_safety_stock_higher_service_level(self):
        """Test higher service level requires more safety stock."""
        optimizer = SparePartsOptimizer()

        ss_95 = optimizer._calculate_safety_stock(
            daily_demand=1.0,
            demand_std=0.3,
            lead_time_days=14,
            lead_time_std=2,
            service_level=0.95
        )

        ss_99 = optimizer._calculate_safety_stock(
            daily_demand=1.0,
            demand_std=0.3,
            lead_time_days=14,
            lead_time_std=2,
            service_level=0.99
        )

        assert ss_99.safety_stock > ss_95.safety_stock

    def test_stocking_strategy_critical_always_stock(self):
        """Test critical parts always get STOCK strategy."""
        optimizer = SparePartsOptimizer()

        strategy, rationale = optimizer._determine_stocking_strategy(
            criticality=PartCriticality.CRITICAL,
            demand_pattern=DemandPattern.SLOW_MOVING,
            unit_cost=10000,
            annual_demand=0.1,
            lead_time_days=7
        )

        assert strategy == StockingStrategy.STOCK
        assert "Critical" in rationale

    def test_stocking_strategy_long_lead_time(self):
        """Test long lead time triggers stocking."""
        optimizer = SparePartsOptimizer()

        strategy, rationale = optimizer._determine_stocking_strategy(
            criticality=PartCriticality.IMPORTANT,
            demand_pattern=DemandPattern.STEADY,
            unit_cost=100,
            annual_demand=10,
            lead_time_days=90  # Very long lead time
        )

        assert strategy == StockingStrategy.STOCK
        assert "lead time" in rationale.lower()

    def test_demand_analysis_from_annual(self):
        """Test demand analysis using annual usage."""
        optimizer = SparePartsOptimizer()

        result = optimizer._analyze_demand(
            usage_history=[],
            annual_usage=120,  # 10 per month
            mtbf_hours=None,
            equipment_count=1
        )

        assert result.annual_demand == 120
        assert result.avg_monthly_demand == 10
        assert result.demand_pattern == DemandPattern.STEADY

    def test_demand_analysis_from_mtbf(self):
        """Test demand analysis using MTBF."""
        optimizer = SparePartsOptimizer()

        result = optimizer._analyze_demand(
            usage_history=[],
            annual_usage=None,
            mtbf_hours=8760,  # 1 failure per year per unit
            equipment_count=5
        )

        assert result.annual_demand == pytest.approx(5, rel=0.1)
        assert result.demand_pattern == DemandPattern.INTERMITTENT

    def test_full_optimization_pipeline(self, sample_spare_part_input):
        """Test full optimization pipeline."""
        optimizer = SparePartsOptimizer()
        result = optimizer.optimize(sample_spare_part_input)

        assert result.part_id == "PUMP-SEAL-001"
        assert result.eoq_result.eoq >= 1
        assert result.safety_stock_result.safety_stock >= 0
        assert len(result.recommendations) > 0
        assert result.provenance_hash is not None

    def test_portfolio_optimization(self):
        """Test portfolio-level optimization."""
        optimizer = SparePartsOptimizer()

        parts = [
            SparePartInput(
                part_id=f"PART-{i}",
                part_name=f"Part {i}",
                equipment_type=EquipmentType.PUMP,
                criticality=PartCriticality.STANDARD,
                unit_cost=100 * (i + 1),
                current_stock=5,
                lead_time_days=14,
                annual_usage=12 * (i + 1)
            )
            for i in range(3)
        ]

        result = optimizer.optimize_portfolio(parts)

        assert result["total_parts"] == 3
        assert "optimization_results" in result
        assert len(result["optimization_results"]) == 3

    def test_convenience_functions(self):
        """Test convenience functions."""
        # EOQ
        eoq_result = calculate_eoq(
            annual_demand=120,
            unit_cost=100,
            ordering_cost=50,
            holding_rate=0.25
        )
        assert eoq_result.eoq > 0

        # Safety stock
        ss_result = calculate_safety_stock(
            daily_demand=0.5,
            demand_std=0.2,
            lead_time_days=14,
            service_level=0.95
        )
        assert ss_result.safety_stock >= 0

        # Stocking strategy
        strategy, rationale = recommend_stocking_strategy(
            criticality="critical",
            unit_cost=1000,
            annual_demand=5,
            lead_time_days=21
        )
        assert strategy == "stock"


# ============================================================================
# TASK-110: Production Impact Modeling Tests
# ============================================================================

class TestProductionImpactModeler:
    """Tests for ProductionImpactModeler class."""

    def test_init_default_precision(self):
        """Test initialization with default precision."""
        modeler = ProductionImpactModeler()
        assert modeler.precision == 2

    def test_triangular_sample_bounds(self):
        """Test triangular sampling stays within bounds."""
        modeler = ProductionImpactModeler()
        import random
        rng = random.Random(42)

        samples = [
            modeler._triangular_sample(1, 3, 5, rng)
            for _ in range(100)
        ]

        assert all(1 <= s <= 5 for s in samples)
        mean = sum(samples) / len(samples)
        assert 2 < mean < 4  # Should be around 3 (mode)

    def test_poisson_sample_mean(self):
        """Test Poisson sampling has correct mean."""
        modeler = ProductionImpactModeler()
        import random
        rng = random.Random(42)

        samples = [modeler._poisson_sample(5, rng) for _ in range(1000)]

        mean = sum(samples) / len(samples)
        assert 4 < mean < 6  # Should be around 5

    def test_severity_classification(self):
        """Test severity classification thresholds."""
        modeler = ProductionImpactModeler()

        assert modeler._classify_severity(0) == ImpactSeverity.NEGLIGIBLE
        assert modeler._classify_severity(3) == ImpactSeverity.MINOR
        assert modeler._classify_severity(15) == ImpactSeverity.MODERATE
        assert modeler._classify_severity(50) == ImpactSeverity.MAJOR
        assert modeler._classify_severity(100) == ImpactSeverity.CATASTROPHIC

    def test_lost_production_cost(self, sample_production_params):
        """Test lost production cost calculation."""
        modeler = ProductionImpactModeler()

        cost = modeler._calculate_lost_production_cost(
            duration_hours=8,
            production=sample_production_params,
            capacity_loss_pct=100
        )

        # 8 hours * 25 units/hr * 0.85 utilization * 100% loss * $150/unit
        expected = 8 * 25 * 0.85 * 1.0 * 150
        assert cost == pytest.approx(expected, rel=0.01)

    def test_quality_loss_cost(self, sample_production_params):
        """Test quality loss cost calculation."""
        modeler = ProductionImpactModeler()

        cost = modeler._calculate_quality_loss_cost(
            duration_hours=8,
            production=sample_production_params,
            quality_impact_pct=10
        )

        assert cost > 0  # Should have some quality cost

    def test_monte_carlo_determinism(
        self,
        sample_failure_scenario,
        sample_production_params,
        sample_equipment_profile
    ):
        """Test Monte Carlo is deterministic with same seed."""
        modeler = ProductionImpactModeler()

        scenarios = [sample_failure_scenario]
        equipment = [sample_equipment_profile]

        result1 = modeler._run_monte_carlo(
            scenarios,
            sample_production_params,
            equipment,
            iterations=1000,
            analysis_period_years=1.0,
            random_seed=42
        )

        result2 = modeler._run_monte_carlo(
            scenarios,
            sample_production_params,
            equipment,
            iterations=1000,
            analysis_period_years=1.0,
            random_seed=42
        )

        assert result1.mean_annual_cost == result2.mean_annual_cost
        assert result1.percentile_95 == result2.percentile_95

    def test_monte_carlo_different_seeds(
        self,
        sample_failure_scenario,
        sample_production_params,
        sample_equipment_profile
    ):
        """Test different seeds produce different results."""
        modeler = ProductionImpactModeler()

        scenarios = [sample_failure_scenario]
        equipment = [sample_equipment_profile]

        result1 = modeler._run_monte_carlo(
            scenarios,
            sample_production_params,
            equipment,
            iterations=1000,
            analysis_period_years=1.0,
            random_seed=42
        )

        result2 = modeler._run_monte_carlo(
            scenarios,
            sample_production_params,
            equipment,
            iterations=1000,
            analysis_period_years=1.0,
            random_seed=123
        )

        # Results should be close but not identical
        assert result1.mean_annual_cost != result2.mean_annual_cost

    def test_full_analysis_pipeline(
        self,
        sample_failure_scenario,
        sample_production_params,
        sample_equipment_profile
    ):
        """Test full analysis pipeline."""
        modeler = ProductionImpactModeler()

        input_data = ProductionImpactInput(
            analysis_name="Test Analysis",
            equipment=[sample_equipment_profile],
            production=sample_production_params,
            scenarios=[sample_failure_scenario],
            analysis_period_years=1.0,
            monte_carlo_iterations=1000,
            random_seed=42
        )

        result = modeler.analyze(input_data)

        assert result.analysis_name == "Test Analysis"
        assert result.expected_annual_cost > 0
        assert len(result.scenario_results) == 1
        assert result.monte_carlo_result.iterations == 1000
        assert result.provenance_hash is not None
        assert len(result.recommendations) > 0

    def test_quick_downtime_cost(self):
        """Test quick downtime cost calculation."""
        modeler = ProductionImpactModeler()

        cost = modeler.quick_downtime_cost(
            downtime_hours=8,
            hourly_production_rate=10,
            production_value_per_unit=100,
            repair_cost=1000
        )

        # 8 * 10 * 100 + 1000 = 9000
        assert cost == 9000

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Downtime cost
        result = estimate_downtime_cost(
            downtime_hours=8,
            hourly_production_value=5000,
            repair_cost=2500
        )
        assert result['total_cost'] == 8 * 5000 + 2500

        # Availability
        avail = calculate_availability(mtbf_hours=1000, mttr_hours=4)
        assert 0.99 < avail['availability'] < 1.0
        assert avail['failure_rate_per_year'] == pytest.approx(8.76, rel=0.01)

        # RPN
        rpn = calculate_risk_priority_number(
            probability=0.1,
            severity=8,
            detectability=6
        )
        assert rpn['rpn'] == 8 * 6 * 6  # occurrence=6 for p=0.1

    def test_recommendations_generated(
        self,
        sample_failure_scenario,
        sample_production_params,
        sample_equipment_profile
    ):
        """Test recommendations are generated."""
        modeler = ProductionImpactModeler()

        input_data = ProductionImpactInput(
            analysis_name="Recommendations Test",
            equipment=[sample_equipment_profile],
            production=sample_production_params,
            scenarios=[sample_failure_scenario],
            monte_carlo_iterations=100,
            random_seed=42
        )

        result = modeler.analyze(input_data)

        assert len(result.recommendations) > 0
        # Should have at least priority recommendation
        assert any("PRIORITY" in r or "priority" in r.lower() for r in result.recommendations)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for predictive models."""

    def test_spare_parts_with_fuel_costs(self, sample_spare_part_input):
        """Test spare parts optimization considers costs properly."""
        optimizer = SparePartsOptimizer()
        result = optimizer.optimize(sample_spare_part_input)

        # Annual costs should be positive
        assert result.current_annual_cost >= 0
        assert result.optimized_annual_cost >= 0

    def test_production_impact_multiple_scenarios(
        self,
        sample_production_params,
        sample_equipment_profile
    ):
        """Test production impact with multiple scenarios."""
        modeler = ProductionImpactModeler()

        scenarios = [
            FailureScenario(
                scenario_id=f"FS-{i}",
                scenario_name=f"Scenario {i}",
                failure_mode=FailureMode.MECHANICAL,
                probability=0.1 * (i + 1),
                duration_hours_min=2,
                duration_hours_max=8,
                capacity_loss_pct=50 * (i + 1),
                repair_cost_min=1000,
                repair_cost_max=3000
            )
            for i in range(3)
        ]

        input_data = ProductionImpactInput(
            analysis_name="Multi-Scenario Test",
            equipment=[sample_equipment_profile],
            production=sample_production_params,
            scenarios=scenarios,
            monte_carlo_iterations=500,
            random_seed=42
        )

        result = modeler.analyze(input_data)

        assert len(result.scenario_results) == 3
        assert result.expected_annual_cost > 0
        # Higher probability scenarios should have higher expected cost
        sorted_results = sorted(
            result.scenario_results,
            key=lambda x: x.expected_cost,
            reverse=True
        )
        assert sorted_results[0].scenario_id == "FS-2"  # Highest probability

    def test_provenance_hashes_unique(self, sample_fuel_prices, sample_spare_part_input):
        """Test provenance hashes are unique across different analyses."""
        fuel_predictor = FuelPricePredictor()
        parts_optimizer = SparePartsOptimizer()

        fuel_input = FuelPriceInput(
            fuel_type=FuelType.NATURAL_GAS,
            historical_prices=sample_fuel_prices,
            forecast_horizon=ForecastHorizon.DAYS_30
        )

        fuel_result = fuel_predictor.predict(fuel_input)
        parts_result = parts_optimizer.optimize(sample_spare_part_input)

        # Different analyses should have different hashes
        assert fuel_result.provenance_hash != parts_result.provenance_hash


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for predictive models."""

    def test_fuel_price_minimum_data(self):
        """Test fuel price prediction with minimum data (30 points)."""
        predictor = FuelPricePredictor()

        base_date = datetime(2024, 1, 1)
        prices = [
            FuelPriceDataPoint(
                date=base_date + timedelta(days=i),
                price=3.0 + i * 0.01,
                unit="MMBtu"
            )
            for i in range(30)  # Minimum required
        ]

        input_data = FuelPriceInput(
            fuel_type=FuelType.NATURAL_GAS,
            historical_prices=prices,
            forecast_horizon=ForecastHorizon.DAYS_7
        )

        result = predictor.predict(input_data)
        assert len(result.forecasts) == 7

    def test_spare_parts_zero_current_stock(self):
        """Test spare parts with zero current stock."""
        optimizer = SparePartsOptimizer()

        input_data = SparePartInput(
            part_id="ZERO-STOCK",
            part_name="Empty Stock Part",
            equipment_type=EquipmentType.VALVE,
            criticality=PartCriticality.ESSENTIAL,
            unit_cost=500,
            current_stock=0,  # Zero stock
            lead_time_days=14,
            annual_usage=12
        )

        result = optimizer.optimize(input_data)

        # Should recommend increasing stock
        assert result.stock_adjustment > 0

    def test_production_impact_zero_probability_scenarios(
        self,
        sample_production_params,
        sample_equipment_profile
    ):
        """Test production impact with very low probability scenarios."""
        modeler = ProductionImpactModeler()

        scenario = FailureScenario(
            scenario_id="LOW-PROB",
            scenario_name="Very Rare Event",
            failure_mode=FailureMode.STRUCTURAL,
            probability=0.001,  # Very low
            duration_hours_min=24,
            duration_hours_max=168,
            capacity_loss_pct=100,
            repair_cost_min=50000,
            repair_cost_max=100000
        )

        input_data = ProductionImpactInput(
            analysis_name="Low Probability Test",
            equipment=[sample_equipment_profile],
            production=sample_production_params,
            scenarios=[scenario],
            monte_carlo_iterations=1000,
            random_seed=42
        )

        result = modeler.analyze(input_data)

        # Expected cost should be low but not zero
        assert result.expected_annual_cost >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
