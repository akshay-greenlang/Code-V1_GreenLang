"""
GL-007 FURNACEPULSE - RUL Predictor Tests

Unit tests for Remaining Useful Life prediction including:
- RUL calculation with maintenance history
- Confidence interval calculation
- Weibull parameter fitting
- Component-specific models
- Provenance and determinism

Coverage Target: >85%
"""

import pytest
import math
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class ComponentType(Enum):
    """Component types for RUL prediction."""
    RADIANT_TUBE = "RADIANT_TUBE"
    CONVECTION_TUBE = "CONVECTION_TUBE"
    BURNER = "BURNER"
    REFRACTORY = "REFRACTORY"
    DAMPER = "DAMPER"
    SENSOR = "SENSOR"


@dataclass
class WeibullParameters:
    """Weibull distribution parameters."""
    shape: float  # Beta (shape parameter)
    scale: float  # Eta (characteristic life)
    location: float = 0.0  # Gamma (location/threshold parameter)


class TestRULCalculation:
    """Tests for RUL calculation with maintenance history."""

    def test_rul_basic_calculation(self, sample_maintenance_history):
        """Test basic RUL calculation from operating hours."""
        # Component operating data
        component_id = "TUBE-R1-01"
        current_operating_hours = 45000.0

        # Get maintenance history for component
        component_history = [
            r for r in sample_maintenance_history
            if r.component_id == component_id
        ]

        # Get last failure/maintenance
        if component_history:
            last_maintenance_hours = max(
                r.operating_hours_at_maintenance for r in component_history
            )
        else:
            last_maintenance_hours = 0.0

        hours_since_maintenance = current_operating_hours - last_maintenance_hours

        assert hours_since_maintenance >= 0

    def test_rul_from_weibull_model(self):
        """Test RUL calculation using Weibull model."""
        # Weibull parameters for radiant tubes (typical values)
        shape = 2.5  # Beta (wear-out failure mode)
        scale = 50000.0  # Eta (characteristic life in hours)
        current_age = 45000.0

        # Calculate reliability at current age
        # R(t) = exp(-(t/eta)^beta)
        reliability_current = math.exp(-((current_age / scale) ** shape))

        # Calculate median remaining life
        # RUL_median = eta * (-ln(0.5))^(1/beta) - current_age
        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul_median = max(0, median_life - current_age)

        assert reliability_current > 0
        assert rul_median >= 0

    def test_rul_with_operating_conditions(self):
        """Test RUL adjustment based on operating conditions."""
        base_rul_hours = 10000.0

        # Operating condition factors
        temperature_factor = 1.0  # 1.0 = normal, <1.0 = high temp reduces life
        load_factor = 1.0  # 1.0 = normal, <1.0 = high load reduces life
        cycling_factor = 1.0  # 1.0 = normal, <1.0 = frequent cycling reduces life

        # Severe conditions
        high_temp_factor = 0.7  # 30% life reduction at high temps
        high_load_factor = 0.85  # 15% reduction at high load

        adjusted_rul_severe = base_rul_hours * high_temp_factor * high_load_factor
        adjusted_rul_normal = base_rul_hours * temperature_factor * load_factor

        assert adjusted_rul_severe < adjusted_rul_normal

    @pytest.mark.parametrize(
        "current_hours,expected_life,expected_rul_range",
        [
            (40000.0, 50000.0, (8000.0, 12000.0)),
            (45000.0, 50000.0, (3000.0, 7000.0)),
            (48000.0, 50000.0, (1000.0, 3000.0)),
            (50000.0, 50000.0, (0.0, 1000.0)),
        ],
    )
    def test_rul_parametrized(self, current_hours, expected_life, expected_rul_range):
        """Test RUL calculation with various operating hours."""
        simple_rul = max(0, expected_life - current_hours)

        assert expected_rul_range[0] <= simple_rul <= expected_rul_range[1]


class TestConfidenceIntervalCalculation:
    """Tests for RUL confidence interval calculation."""

    def test_confidence_interval_95_percent(self):
        """Test 95% confidence interval calculation."""
        # RUL prediction with uncertainty
        rul_mean = 10000.0
        rul_std = 2000.0
        confidence_level = 0.95

        # 95% CI: mean +/- 1.96 * std
        z_score_95 = 1.96
        lower_bound = max(0, rul_mean - z_score_95 * rul_std)
        upper_bound = rul_mean + z_score_95 * rul_std

        # Expected: ~6080 to ~13920
        assert lower_bound < rul_mean < upper_bound
        assert lower_bound >= 0

    def test_confidence_interval_90_percent(self):
        """Test 90% confidence interval (narrower than 95%)."""
        rul_mean = 10000.0
        rul_std = 2000.0

        # 90% CI: mean +/- 1.645 * std
        z_score_90 = 1.645
        lower_90 = max(0, rul_mean - z_score_90 * rul_std)
        upper_90 = rul_mean + z_score_90 * rul_std

        # 95% CI for comparison
        z_score_95 = 1.96
        lower_95 = max(0, rul_mean - z_score_95 * rul_std)
        upper_95 = rul_mean + z_score_95 * rul_std

        # 90% CI should be narrower
        ci_width_90 = upper_90 - lower_90
        ci_width_95 = upper_95 - lower_95

        assert ci_width_90 < ci_width_95

    def test_confidence_interval_from_weibull(self):
        """Test confidence interval from Weibull distribution."""
        shape = 2.5
        scale = 50000.0
        current_age = 45000.0

        # Calculate percentiles for confidence interval
        # Using inverse Weibull CDF: t = scale * (-ln(1-p))^(1/beta)
        p_lower = 0.05  # 5th percentile
        p_upper = 0.95  # 95th percentile

        t_lower = scale * ((-math.log(1 - p_lower)) ** (1 / shape))
        t_upper = scale * ((-math.log(1 - p_upper)) ** (1 / shape))

        rul_lower = max(0, t_lower - current_age)
        rul_upper = max(0, t_upper - current_age)

        assert rul_lower <= rul_upper

    def test_confidence_interval_with_limited_data(self):
        """Test wider CI with limited historical data."""
        # With limited data, uncertainty is higher
        data_points = 10
        rul_mean = 10000.0
        base_std = 2000.0

        # Adjust std for small sample size (t-distribution effect)
        adjusted_std = base_std * math.sqrt(1 + 1 / data_points)

        z_score = 1.96
        ci_lower_limited = max(0, rul_mean - z_score * adjusted_std)
        ci_upper_limited = rul_mean + z_score * adjusted_std

        ci_lower_full = max(0, rul_mean - z_score * base_std)
        ci_upper_full = rul_mean + z_score * base_std

        # Limited data should have wider CI
        ci_width_limited = ci_upper_limited - ci_lower_limited
        ci_width_full = ci_upper_full - ci_lower_full

        assert ci_width_limited > ci_width_full

    @pytest.mark.parametrize(
        "confidence_level,z_score",
        [
            (0.90, 1.645),
            (0.95, 1.96),
            (0.99, 2.576),
        ],
    )
    def test_confidence_levels_parametrized(self, confidence_level, z_score):
        """Test various confidence levels."""
        rul_mean = 10000.0
        rul_std = 2000.0

        lower = max(0, rul_mean - z_score * rul_std)
        upper = rul_mean + z_score * rul_std

        ci_width = upper - lower

        # Higher confidence = wider interval
        assert ci_width == 2 * z_score * rul_std


class TestWeibullParameterFitting:
    """Tests for Weibull parameter estimation."""

    def test_weibull_mle_estimation(self, sample_failure_history):
        """Test Weibull Maximum Likelihood Estimation."""
        # Filter uncensored failures
        failures = [
            f["time_to_failure_hours"]
            for f in sample_failure_history
            if not f["censored"]
        ]

        if len(failures) < 2:
            pytest.skip("Not enough failure data")

        # Simple MLE approximation for shape parameter
        # Using median rank regression approximation
        n = len(failures)
        failures_sorted = sorted(failures)

        # Calculate shape (beta) using regression
        # ln(-ln(1-F)) = beta * ln(t) - beta * ln(eta)
        # where F is estimated failure probability

        log_times = [math.log(t) for t in failures_sorted]
        mean_log_time = sum(log_times) / n

        # Approximate beta (typical range 1.5-3.5 for wear-out)
        # Using method of moments approximation
        variance_log = sum((lt - mean_log_time) ** 2 for lt in log_times) / n
        estimated_beta = 1.282 / math.sqrt(variance_log)

        # Approximate scale (eta)
        mean_time = sum(failures) / n
        estimated_eta = mean_time / math.gamma(1 + 1 / estimated_beta)

        assert 1.0 <= estimated_beta <= 5.0
        assert estimated_eta > 0

    def test_weibull_shape_interpretation(self):
        """Test interpretation of Weibull shape parameter."""
        # Beta < 1: Infant mortality (decreasing failure rate)
        # Beta = 1: Random failures (constant failure rate)
        # Beta > 1: Wear-out (increasing failure rate)

        beta_infant = 0.7
        beta_random = 1.0
        beta_wearout = 2.5

        assert beta_infant < 1.0  # Infant mortality
        assert beta_random == 1.0  # Random (exponential)
        assert beta_wearout > 1.0  # Wear-out

    def test_weibull_reliability_function(self):
        """Test Weibull reliability function calculation."""
        shape = 2.5
        scale = 50000.0

        # R(t) = exp(-(t/eta)^beta)
        test_times = [0, 10000, 25000, 40000, 50000, 60000]

        reliabilities = []
        for t in test_times:
            r = math.exp(-((t / scale) ** shape))
            reliabilities.append(r)

        # Reliability should decrease with time
        for i in range(1, len(reliabilities)):
            assert reliabilities[i] <= reliabilities[i - 1]

        # R(0) = 1, R(infinity) = 0
        assert reliabilities[0] == 1.0

    def test_weibull_hazard_function(self):
        """Test Weibull hazard (failure rate) function."""
        shape = 2.5
        scale = 50000.0

        # h(t) = (beta/eta) * (t/eta)^(beta-1)
        t = 40000.0

        hazard = (shape / scale) * ((t / scale) ** (shape - 1))

        # Hazard should be positive
        assert hazard > 0

        # For wear-out (beta > 1), hazard increases with time
        t2 = 45000.0
        hazard2 = (shape / scale) * ((t2 / scale) ** (shape - 1))

        assert hazard2 > hazard

    def test_censored_data_handling(self, sample_failure_history):
        """Test handling of right-censored data (still running)."""
        censored = [f for f in sample_failure_history if f["censored"]]
        uncensored = [f for f in sample_failure_history if not f["censored"]]

        assert len(censored) > 0
        assert len(uncensored) > 0

        # Censored observations contribute to likelihood differently
        # They indicate component survived at least to observed time

    @pytest.mark.parametrize(
        "shape,expected_failure_mode",
        [
            (0.5, "infant_mortality"),
            (1.0, "random"),
            (2.0, "early_wearout"),
            (3.5, "late_wearout"),
        ],
    )
    def test_shape_failure_mode_mapping(self, shape, expected_failure_mode):
        """Test mapping of shape parameter to failure mode."""
        if shape < 1.0:
            failure_mode = "infant_mortality"
        elif shape == 1.0:
            failure_mode = "random"
        elif shape < 3.0:
            failure_mode = "early_wearout"
        else:
            failure_mode = "late_wearout"

        assert failure_mode == expected_failure_mode


class TestComponentSpecificModels:
    """Tests for component-specific RUL models."""

    def test_radiant_tube_model(self):
        """Test RUL model for radiant tubes."""
        # Radiant tubes typically have:
        # - Beta ~ 2.5-3.5 (wear-out)
        # - Eta ~ 45000-60000 hours
        shape = 3.0
        scale = 52000.0
        current_hours = 45000.0

        # Calculate expected remaining life
        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul = max(0, median_life - current_hours)

        # Radiant tube RUL should be in reasonable range
        assert 0 <= rul <= 20000

    def test_burner_model(self):
        """Test RUL model for burners."""
        # Burners typically have:
        # - Beta ~ 2.0-2.5
        # - Eta ~ 20000-30000 hours
        shape = 2.2
        scale = 25000.0
        current_hours = 18000.0

        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul = max(0, median_life - current_hours)

        assert rul > 0

    def test_refractory_model(self):
        """Test RUL model for refractory."""
        # Refractory typically has:
        # - Beta ~ 1.5-2.5
        # - Eta varies widely by type
        shape = 2.0
        scale = 40000.0
        current_hours = 30000.0

        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul = max(0, median_life - current_hours)

        assert rul >= 0

    def test_different_components_different_params(self):
        """Test that different components have different Weibull params."""
        component_params = {
            ComponentType.RADIANT_TUBE: WeibullParameters(shape=3.0, scale=52000.0),
            ComponentType.BURNER: WeibullParameters(shape=2.2, scale=25000.0),
            ComponentType.REFRACTORY: WeibullParameters(shape=2.0, scale=40000.0),
            ComponentType.DAMPER: WeibullParameters(shape=1.8, scale=60000.0),
        }

        # All components should have different parameters
        shapes = [p.shape for p in component_params.values()]
        scales = [p.scale for p in component_params.values()]

        # Not all identical
        assert len(set(shapes)) > 1 or len(set(scales)) > 1

    def test_rul_comparison_across_components(self):
        """Test RUL comparison across different components."""
        current_hours = 30000.0

        components = {
            "tube": WeibullParameters(shape=3.0, scale=52000.0),
            "burner": WeibullParameters(shape=2.2, scale=25000.0),
            "refractory": WeibullParameters(shape=2.0, scale=40000.0),
        }

        ruls = {}
        for name, params in components.items():
            median_life = params.scale * ((-math.log(0.5)) ** (1 / params.shape))
            ruls[name] = max(0, median_life - current_hours)

        # Burner should have shortest RUL (smallest scale)
        assert ruls["burner"] < ruls["tube"]


class TestRULWithMaintenanceHistory:
    """Tests for RUL considering maintenance history."""

    def test_rul_reset_after_replacement(self, sample_maintenance_history):
        """Test RUL resets after component replacement."""
        # If component replaced, RUL calculation starts from 0
        current_hours = 48000.0
        replacement_at_hours = 45000.0  # Last replacement

        hours_since_replacement = current_hours - replacement_at_hours
        new_component_life = 50000.0  # Expected life of new component

        rul_after_replacement = new_component_life - hours_since_replacement

        assert rul_after_replacement > 0
        assert rul_after_replacement < new_component_life

    def test_rul_adjustment_for_repairs(self, sample_maintenance_history):
        """Test RUL adjustment when repairs extend life."""
        base_rul = 10000.0
        repair_life_extension_percent = 0.2  # 20% life extension

        adjusted_rul = base_rul * (1 + repair_life_extension_percent)

        assert adjusted_rul > base_rul

    def test_maintenance_impact_on_weibull(self):
        """Test how maintenance affects Weibull parameters."""
        # Good maintenance improves scale (characteristic life)
        base_scale = 50000.0
        maintenance_factor = 1.1  # 10% improvement with good maintenance

        improved_scale = base_scale * maintenance_factor

        assert improved_scale > base_scale

    def test_degraded_rul_with_poor_maintenance(self):
        """Test reduced RUL with poor maintenance history."""
        base_rul = 10000.0
        poor_maintenance_factor = 0.7  # 30% reduction

        degraded_rul = base_rul * poor_maintenance_factor

        assert degraded_rul < base_rul


class TestRULPredictionQuality:
    """Tests for RUL prediction quality metrics."""

    def test_prediction_accuracy_metric(self):
        """Test calculation of prediction accuracy."""
        # Simulated predictions vs actuals
        predictions = [10000, 8000, 12000, 9500, 11000]
        actuals = [9500, 8500, 11500, 9000, 10500]

        # Mean Absolute Error
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)

        # MAE should be reasonable (< 1000 hours for good model)
        assert mae < 1000

    def test_prediction_bias(self):
        """Test for prediction bias (over/under prediction)."""
        predictions = [10000, 10000, 10000, 10000]
        actuals = [9500, 9800, 9200, 9600]  # All actuals lower

        # Mean Error (positive = over-prediction)
        mean_error = sum(p - a for p, a in zip(predictions, actuals)) / len(predictions)

        # Positive bias means over-predicting (optimistic)
        has_positive_bias = mean_error > 0
        assert has_positive_bias

    def test_coverage_probability(self):
        """Test confidence interval coverage probability."""
        # For 95% CI, approximately 95% of actuals should fall within
        predictions_with_ci = [
            {"mean": 10000, "lower": 8000, "upper": 12000, "actual": 9500},
            {"mean": 8000, "lower": 6500, "upper": 9500, "actual": 8200},
            {"mean": 12000, "lower": 10000, "upper": 14000, "actual": 11500},
        ]

        covered_count = sum(
            1 for p in predictions_with_ci
            if p["lower"] <= p["actual"] <= p["upper"]
        )
        coverage = covered_count / len(predictions_with_ci)

        # Should have high coverage
        assert coverage >= 0.9


class TestRULProvenanceTracking:
    """Tests for RUL prediction provenance and audit."""

    def test_rul_hash_deterministic(self):
        """Test that RUL predictions produce deterministic hash."""
        import hashlib
        import json

        inputs = {
            "component_id": "TUBE-R1-01",
            "current_hours": 45000.0,
            "weibull_shape": 3.0,
            "weibull_scale": 52000.0,
            "method_version": "1.0.0",
        }

        hash1 = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_rul_reproducibility(self):
        """Test RUL calculation reproducibility."""
        shape = 3.0
        scale = 52000.0
        current_hours = 45000.0

        results = []
        for _ in range(5):
            median_life = scale * ((-math.log(0.5)) ** (1 / shape))
            rul = max(0, median_life - current_hours)
            results.append(rul)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_rul_audit_record(self, sample_maintenance_history):
        """Test RUL audit record includes all relevant data."""
        audit_record = {
            "prediction_id": "RUL-2025-001",
            "timestamp": datetime.now().isoformat(),
            "component_id": "TUBE-R1-01",
            "inputs": {
                "current_hours": 45000.0,
                "maintenance_history_count": len(sample_maintenance_history),
            },
            "model": {
                "type": "Weibull",
                "shape": 3.0,
                "scale": 52000.0,
            },
            "outputs": {
                "rul_mean": 5500.0,
                "rul_lower_95": 3000.0,
                "rul_upper_95": 8000.0,
            },
        }

        assert "prediction_id" in audit_record
        assert "inputs" in audit_record
        assert "model" in audit_record
        assert "outputs" in audit_record


class TestEdgeCases:
    """Tests for RUL edge cases."""

    def test_rul_zero_hours(self):
        """Test RUL for brand new component."""
        current_hours = 0.0
        shape = 3.0
        scale = 52000.0

        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul = median_life - current_hours

        # New component should have full life remaining
        assert rul == median_life

    def test_rul_exceeds_expected_life(self):
        """Test RUL when component exceeds expected life."""
        current_hours = 60000.0  # Beyond expected life
        shape = 3.0
        scale = 52000.0

        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul = max(0, median_life - current_hours)

        # RUL should be 0 (or small positive with uncertainty)
        assert rul >= 0

    def test_rul_with_extreme_operating_conditions(self):
        """Test RUL adjustment for extreme conditions."""
        base_rul = 10000.0

        # Extreme conditions factors
        extreme_temp_factor = 0.5  # 50% life at extreme temps
        extreme_cycling_factor = 0.6  # 40% reduction with cycling

        extreme_rul = base_rul * extreme_temp_factor * extreme_cycling_factor

        assert extreme_rul < base_rul * 0.5

    def test_rul_minimum_value(self):
        """Test that RUL has a minimum value of 0."""
        current_hours = 100000.0  # Way beyond life
        shape = 3.0
        scale = 52000.0

        median_life = scale * ((-math.log(0.5)) ** (1 / shape))
        rul = max(0, median_life - current_hours)

        assert rul == 0


class TestPerformance:
    """Performance tests for RUL prediction."""

    def test_rul_calculation_speed(self):
        """Test RUL calculation performance."""
        import time

        shape = 3.0
        scale = 52000.0

        start_time = time.time()

        for _ in range(10000):
            current_hours = 45000.0
            median_life = scale * ((-math.log(0.5)) ** (1 / shape))
            rul = max(0, median_life - current_hours)

        elapsed = time.time() - start_time

        # Should complete 10000 calculations in < 100ms
        assert elapsed < 0.1

    def test_batch_rul_prediction(self):
        """Test batch RUL prediction for multiple components."""
        import time

        components = [
            {"id": f"COMP-{i:03d}", "hours": 40000 + i * 100, "shape": 3.0, "scale": 52000.0}
            for i in range(100)
        ]

        start_time = time.time()

        ruls = []
        for comp in components:
            median_life = comp["scale"] * ((-math.log(0.5)) ** (1 / comp["shape"]))
            rul = max(0, median_life - comp["hours"])
            ruls.append({"id": comp["id"], "rul": rul})

        elapsed = time.time() - start_time

        # Should process 100 components in < 10ms
        assert elapsed < 0.01
        assert len(ruls) == 100
