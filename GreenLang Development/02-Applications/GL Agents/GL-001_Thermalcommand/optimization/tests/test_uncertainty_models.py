"""
Tests for Uncertainty Models

Tests for deterministic uncertainty quantification including:
    - Deterministic RNG reproducibility
    - Prediction interval generation
    - Quantile set generation
    - Scenario generation
    - Weather, price, and demand uncertainty models

All tests verify bit-perfect reproducibility and provenance tracking.

Author: GreenLang Process Heat Team
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ..uncertainty_models import (
    DeterministicRNG,
    UncertaintyModelEngine,
    WeatherUncertaintyModel,
    PriceUncertaintyModel,
    DemandUncertaintyModel,
)
from ..uq_schemas import (
    UncertaintySource,
    UncertaintySourceType,
    DistributionType,
)


class TestDeterministicRNG:
    """Tests for DeterministicRNG bit-perfect reproducibility."""

    def test_same_seed_same_sequence(self):
        """Same seed should produce identical sequence."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=42)

        for _ in range(100):
            assert rng1.next_int() == rng2.next_int()

    def test_different_seeds_different_sequence(self):
        """Different seeds should produce different sequences."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=43)

        # Very unlikely to be equal
        assert rng1.next_int() != rng2.next_int()

    def test_reset_restarts_sequence(self):
        """Reset should restart sequence from beginning."""
        rng = DeterministicRNG(seed=42)

        first_values = [rng.next_int() for _ in range(10)]

        rng.reset()

        second_values = [rng.next_int() for _ in range(10)]

        assert first_values == second_values

    def test_uniform_in_range(self):
        """Uniform values should be in [0, 1)."""
        rng = DeterministicRNG(seed=42)

        for _ in range(1000):
            u = rng.next_uniform()
            assert Decimal("0") <= u < Decimal("1")

    def test_normal_distribution(self):
        """Normal distribution should have reasonable mean and std."""
        rng = DeterministicRNG(seed=42)

        samples = [rng.next_normal() for _ in range(1000)]

        mean = sum(samples) / Decimal("1000")
        variance = sum((x - mean) ** 2 for x in samples) / Decimal("1000")
        std = variance.sqrt()

        # Mean should be close to 0
        assert abs(mean) < Decimal("0.1")
        # Std should be close to 1
        assert Decimal("0.8") < std < Decimal("1.2")

    def test_normal_with_parameters(self):
        """Normal with custom mean and std."""
        rng = DeterministicRNG(seed=42)

        samples = [rng.next_normal(mean=Decimal("100"), std=Decimal("10")) for _ in range(1000)]

        mean = sum(samples) / Decimal("1000")

        # Mean should be close to 100
        assert Decimal("98") < mean < Decimal("102")


class TestUncertaintyModelEngine:
    """Tests for UncertaintyModelEngine determinism and correctness."""

    def test_engine_reproducibility(self):
        """Engine with same seed should produce identical results."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MEASUREMENT,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine1 = UncertaintyModelEngine(seed=42)
        engine2 = UncertaintyModelEngine(seed=42)

        interval1 = engine1.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        interval2 = engine2.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        assert interval1.lower_bound == interval2.lower_bound
        assert interval1.upper_bound == interval2.upper_bound
        assert interval1.provenance.combined_hash == interval2.provenance.combined_hash

    def test_prediction_interval_normal_distribution(self):
        """Normal distribution intervals should be symmetric."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MEASUREMENT,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine = UncertaintyModelEngine(seed=42)

        interval = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        # Interval should be symmetric around point estimate
        lower_distance = interval.point_estimate - interval.lower_bound
        upper_distance = interval.upper_bound - interval.point_estimate

        assert abs(lower_distance - upper_distance) < Decimal("0.001")

    def test_prediction_interval_lognormal_distribution(self):
        """Lognormal distribution intervals should be asymmetric."""
        source = UncertaintySource(
            name="price",
            source_type=UncertaintySourceType.PRICE,
            distribution=DistributionType.LOGNORMAL,
            parameters={"mean": Decimal("100"), "sigma": Decimal("0.20")},
            unit="USD"
        )

        engine = UncertaintyModelEngine(seed=42)

        interval = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        # Upper bound should be farther from point than lower bound (right skew)
        lower_distance = interval.point_estimate - interval.lower_bound
        upper_distance = interval.upper_bound - interval.point_estimate

        assert upper_distance > lower_distance

    def test_prediction_interval_uniform_distribution(self):
        """Uniform distribution intervals should respect bounds."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.PARAMETER,
            distribution=DistributionType.UNIFORM,
            parameters={"half_width": Decimal("10")},
            unit="units"
        )

        engine = UncertaintyModelEngine(seed=42)

        interval = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        assert interval.lower_bound >= Decimal("90")
        assert interval.upper_bound <= Decimal("110")

    def test_prediction_interval_horizon_scaling(self):
        """Uncertainty should increase with horizon."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine = UncertaintyModelEngine(seed=42)

        interval_short = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=30
        )

        # Re-seed for fair comparison
        engine = UncertaintyModelEngine(seed=42)

        interval_long = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=240
        )

        assert interval_long.interval_width > interval_short.interval_width

    def test_prediction_interval_confidence_level(self):
        """Higher confidence should produce wider intervals."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine = UncertaintyModelEngine(seed=42)

        interval_90 = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        engine = UncertaintyModelEngine(seed=42)

        interval_95 = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.95"),
            horizon_minutes=60
        )

        assert interval_95.interval_width > interval_90.interval_width

    def test_prediction_interval_has_provenance(self):
        """Prediction interval should have provenance tracking."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine = UncertaintyModelEngine(seed=42)

        interval = engine.generate_prediction_interval(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            confidence_level=Decimal("0.90"),
            horizon_minutes=60
        )

        assert interval.provenance is not None
        assert interval.provenance.calculation_type == "prediction_interval_generation"
        assert len(interval.provenance.combined_hash) == 64


class TestQuantileGeneration:
    """Tests for quantile set generation."""

    def test_generate_quantile_set(self):
        """Quantile set should be generated correctly."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine = UncertaintyModelEngine(seed=42)

        qset = engine.generate_quantile_set(
            point_estimate=Decimal("100"),
            uncertainty_source=source,
            quantile_probs=[Decimal("0.10"), Decimal("0.50"), Decimal("0.90")],
            horizon_minutes=60
        )

        assert len(qset.quantiles) == 3
        assert qset.p10 is not None
        assert qset.p50 is not None
        assert qset.p90 is not None

        # Quantiles should be ordered
        assert qset.p10 < qset.p50 < qset.p90

    def test_quantile_set_reproducibility(self):
        """Quantile set should be reproducible."""
        source = UncertaintySource(
            name="test",
            source_type=UncertaintySourceType.MODEL,
            distribution=DistributionType.NORMAL,
            parameters={"mean": Decimal("100"), "std": Decimal("10")},
            unit="units"
        )

        engine1 = UncertaintyModelEngine(seed=42)
        engine2 = UncertaintyModelEngine(seed=42)

        qset1 = engine1.generate_quantile_set(
            point_estimate=Decimal("100"),
            uncertainty_source=source
        )

        qset2 = engine2.generate_quantile_set(
            point_estimate=Decimal("100"),
            uncertainty_source=source
        )

        assert qset1.provenance.combined_hash == qset2.provenance.combined_hash
        for q1, q2 in zip(qset1.quantiles, qset2.quantiles):
            assert q1.value == q2.value


class TestScenarioGeneration:
    """Tests for scenario set generation."""

    def test_generate_scenarios(self):
        """Scenario set should be generated correctly."""
        sources = [
            UncertaintySource(
                name="temperature",
                source_type=UncertaintySourceType.WEATHER,
                distribution=DistributionType.NORMAL,
                parameters={"mean": Decimal("25"), "std": Decimal("3")},
                unit="degC"
            ),
            UncertaintySource(
                name="demand",
                source_type=UncertaintySourceType.DEMAND,
                distribution=DistributionType.NORMAL,
                parameters={"mean": Decimal("100"), "std": Decimal("10")},
                unit="MW"
            )
        ]

        engine = UncertaintyModelEngine(seed=42)
        now = datetime.utcnow()

        scenario_set = engine.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=10,
            horizon_start=now,
            horizon_end=now + timedelta(hours=24),
            include_base_case=True,
            include_worst_case=True
        )

        assert scenario_set.num_scenarios == 10

        # Should have base case
        base = scenario_set.get_base_case()
        assert base is not None
        assert base.is_base_case is True

        # Should have worst cases
        worst = scenario_set.get_worst_cases()
        assert len(worst) >= 1

        # All scenarios should have variables
        for scenario in scenario_set.scenarios:
            assert len(scenario.variables) == 2

    def test_scenario_probabilities_sum_to_one(self):
        """Scenario probabilities should sum to 1."""
        sources = [
            UncertaintySource(
                name="test",
                source_type=UncertaintySourceType.MODEL,
                distribution=DistributionType.NORMAL,
                parameters={"mean": Decimal("0"), "std": Decimal("1")},
                unit="units"
            )
        ]

        engine = UncertaintyModelEngine(seed=42)
        now = datetime.utcnow()

        scenario_set = engine.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=20,
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        total_prob = sum(s.probability for s in scenario_set.scenarios)
        assert abs(total_prob - Decimal("1.0")) < Decimal("0.001")

    def test_scenario_generation_reproducibility(self):
        """Scenario generation should be reproducible."""
        sources = [
            UncertaintySource(
                name="test",
                source_type=UncertaintySourceType.MODEL,
                distribution=DistributionType.NORMAL,
                parameters={"mean": Decimal("100"), "std": Decimal("10")},
                unit="units"
            )
        ]

        now = datetime.utcnow()

        engine1 = UncertaintyModelEngine(seed=42)
        scenario_set1 = engine1.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=10,
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        engine2 = UncertaintyModelEngine(seed=42)
        scenario_set2 = engine2.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=10,
            horizon_start=now,
            horizon_end=now + timedelta(hours=1)
        )

        # Compare provenance hashes
        assert scenario_set1.provenance.combined_hash == scenario_set2.provenance.combined_hash

        # Compare scenario values
        for s1, s2 in zip(scenario_set1.scenarios, scenario_set2.scenarios):
            for v1, v2 in zip(s1.variables, s2.variables):
                assert v1.value == v2.value

    def test_correlated_scenario_generation(self):
        """Correlated scenarios should respect correlation matrix."""
        sources = [
            UncertaintySource(
                name="var1",
                source_type=UncertaintySourceType.MODEL,
                distribution=DistributionType.NORMAL,
                parameters={"mean": Decimal("0"), "std": Decimal("1")},
                unit="units"
            ),
            UncertaintySource(
                name="var2",
                source_type=UncertaintySourceType.MODEL,
                distribution=DistributionType.NORMAL,
                parameters={"mean": Decimal("0"), "std": Decimal("1")},
                unit="units"
            )
        ]

        # High positive correlation
        correlation_matrix = [
            [Decimal("1.0"), Decimal("0.9")],
            [Decimal("0.9"), Decimal("1.0")]
        ]

        engine = UncertaintyModelEngine(seed=42)
        now = datetime.utcnow()

        scenario_set = engine.generate_scenarios(
            uncertainty_sources=sources,
            num_scenarios=100,
            horizon_start=now,
            horizon_end=now + timedelta(hours=1),
            include_base_case=False,
            include_worst_case=False,
            correlation_matrix=correlation_matrix
        )

        # Calculate sample correlation
        values1 = [s.get_value("var1") for s in scenario_set.scenarios]
        values2 = [s.get_value("var2") for s in scenario_set.scenarios]

        mean1 = sum(values1) / Decimal("100")
        mean2 = sum(values2) / Decimal("100")

        cov = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2)) / Decimal("100")
        var1 = sum((v - mean1) ** 2 for v in values1) / Decimal("100")
        var2 = sum((v - mean2) ** 2 for v in values2) / Decimal("100")

        sample_corr = cov / (var1.sqrt() * var2.sqrt())

        # Sample correlation should be positive (reflecting the high correlation)
        assert sample_corr > Decimal("0.5")


class TestWeatherUncertaintyModel:
    """Tests for WeatherUncertaintyModel."""

    def test_create_temperature_source(self):
        """Temperature uncertainty source should be created."""
        model = WeatherUncertaintyModel(seed=42)

        source = model.create_temperature_source(
            forecast_temperature_c=Decimal("25.0"),
            forecast_horizon_hours=24
        )

        assert source.name.startswith("temperature")
        assert source.source_type == UncertaintySourceType.WEATHER
        assert source.distribution == DistributionType.NORMAL
        assert source.unit == "degC"

    def test_temperature_uncertainty_grows_with_horizon(self):
        """Temperature uncertainty should grow with forecast horizon."""
        model = WeatherUncertaintyModel(seed=42)

        source_short = model.create_temperature_source(
            forecast_temperature_c=Decimal("25.0"),
            forecast_horizon_hours=6
        )

        source_long = model.create_temperature_source(
            forecast_temperature_c=Decimal("25.0"),
            forecast_horizon_hours=48
        )

        assert source_long.parameters["std"] > source_short.parameters["std"]

    def test_generate_weather_scenarios(self):
        """Weather scenarios should be generated."""
        model = WeatherUncertaintyModel(seed=42)

        scenarios = model.generate_weather_scenarios(
            temperature_c=Decimal("25.0"),
            solar_w_m2=Decimal("500.0"),
            horizon_hours=24,
            num_scenarios=10
        )

        assert scenarios.num_scenarios == 10

        # Check for temperature and solar variables
        for scenario in scenarios.scenarios:
            temp = scenario.get_variable("temperature_default")
            solar = scenario.get_variable("solar_irradiance")
            assert temp is not None or solar is not None


class TestPriceUncertaintyModel:
    """Tests for PriceUncertaintyModel."""

    def test_create_electricity_price_source(self):
        """Electricity price uncertainty source should be created."""
        model = PriceUncertaintyModel(seed=42)

        source = model.create_electricity_price_source(
            forecast_price_usd_kwh=Decimal("0.10"),
            forecast_horizon_hours=24,
            market_volatility="normal"
        )

        assert source.name == "electricity_price"
        assert source.source_type == UncertaintySourceType.PRICE
        assert source.distribution == DistributionType.LOGNORMAL
        assert source.unit == "USD/kWh"

    def test_price_volatility_affects_uncertainty(self):
        """Higher volatility should mean wider uncertainty."""
        model = PriceUncertaintyModel(seed=42)

        source_low = model.create_electricity_price_source(
            forecast_price_usd_kwh=Decimal("0.10"),
            market_volatility="low"
        )

        source_high = model.create_electricity_price_source(
            forecast_price_usd_kwh=Decimal("0.10"),
            market_volatility="high"
        )

        assert source_high.parameters["sigma"] > source_low.parameters["sigma"]


class TestDemandUncertaintyModel:
    """Tests for DemandUncertaintyModel."""

    def test_create_heat_demand_source(self):
        """Heat demand uncertainty source should be created."""
        model = DemandUncertaintyModel(seed=42)

        source = model.create_heat_demand_source(
            forecast_demand_mw=Decimal("100.0"),
            forecast_horizon_hours=24,
            process_type="continuous"
        )

        assert source.name == "heat_demand"
        assert source.source_type == UncertaintySourceType.DEMAND
        assert source.unit == "MW_th"

    def test_batch_process_higher_uncertainty(self):
        """Batch processes should have higher uncertainty than continuous."""
        model = DemandUncertaintyModel(seed=42)

        source_continuous = model.create_heat_demand_source(
            forecast_demand_mw=Decimal("100.0"),
            process_type="continuous"
        )

        source_batch = model.create_heat_demand_source(
            forecast_demand_mw=Decimal("100.0"),
            process_type="batch"
        )

        # Batch has higher std relative to mean
        cv_continuous = source_continuous.parameters["std"] / Decimal("100")
        cv_batch = source_batch.parameters["std"] / Decimal("100")

        assert cv_batch > cv_continuous

    def test_generate_demand_scenarios(self):
        """Demand scenarios should be generated."""
        model = DemandUncertaintyModel(seed=42)

        scenarios = model.generate_demand_scenarios(
            heat_demand_mw=Decimal("100.0"),
            horizon_hours=24,
            num_scenarios=10
        )

        assert scenarios.num_scenarios == 10

        # Check for demand variable
        for scenario in scenarios.scenarios:
            demand = scenario.get_variable("heat_demand")
            assert demand is not None
            assert demand.unit == "MW_th"
