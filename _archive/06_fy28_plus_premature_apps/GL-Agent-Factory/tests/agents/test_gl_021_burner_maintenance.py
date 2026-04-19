"""
Unit Tests for GL-021: Burner Maintenance Predictor Agent (BURNERSENTRY)

Comprehensive test suite covering:
- Weibull reliability calculations (R(t), h(t), MTTF, RUL)
- Flame quality assessment
- Health score calculation
- Maintenance priority determination
- Replacement decision logic
- Degradation rate analysis
- Provenance hash generation

Target: 85%+ code coverage

Reference:
- MIL-HDBK-217F: Reliability Prediction of Electronic Equipment
- IEEE Std 493: Design of Reliable Industrial Systems
- IEC 61649: Weibull Analysis

Run with:
    pytest tests/agents/test_gl_021_burner_maintenance.py -v --cov=backend/agents/gl_021_burner_maintenance
"""

import math
import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from agents.gl_021_burner_maintenance.agent import (
    BurnerMaintenancePredictorAgent,
    BurnerInput,
    BurnerOutput,
    WeibullParameters,
    FlameQualityMetrics,
    BurnerComponentHealth,
    MaintenanceRecommendation,
    MaintenancePriority,
    FuelType,
    AgentConfig,
)

from agents.gl_021_burner_maintenance.calculators.weibull import (
    weibull_reliability,
    weibull_failure_rate,
    weibull_mean_life,
    remaining_useful_life,
    calculate_failure_probability,
    weibull_percentile_life,
    weibull_variance,
    estimate_weibull_parameters,
)


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestBurnerAgentInitialization:
    """Tests for BurnerMaintenancePredictorAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self):
        """Test agent initializes correctly with default config."""
        agent = BurnerMaintenancePredictorAgent()

        assert agent is not None
        assert agent.agent_id == "GL-021"
        assert agent.agent_name == "BURNERSENTRY"
        assert agent.version == "1.0.0"

    @pytest.mark.unit
    def test_agent_initializes_with_custom_config(self):
        """Test agent initializes with custom configuration."""
        config = AgentConfig(
            reliability_threshold=0.6,
            critical_health_threshold=25,
        )
        agent = BurnerMaintenancePredictorAgent(config=config)

        assert agent.config.reliability_threshold == 0.6
        assert agent.config.critical_health_threshold == 25

    @pytest.mark.unit
    def test_default_reliability_threshold(self):
        """Test default reliability threshold is 0.5 (median life)."""
        agent = BurnerMaintenancePredictorAgent()

        assert agent.config.reliability_threshold == 0.5


# =============================================================================
# Test Class: Weibull Reliability Calculations
# =============================================================================


class TestWeibullReliability:
    """Tests for Weibull reliability function R(t)."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_reliability_at_zero_is_one(self, weibull_params_typical):
        """Test R(0) = 1 (100% reliability at start)."""
        R = weibull_reliability(
            t=0,
            beta=weibull_params_typical["beta"],
            eta=weibull_params_typical["eta"],
        )

        assert R == 1.0

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_reliability_at_eta_is_368(self, weibull_params_typical):
        """
        Test R(eta) = e^(-1) = 0.368 (characteristic life).

        The characteristic life (eta) is the time at which 63.2%
        of units will have failed (reliability = 36.8%).
        """
        R = weibull_reliability(
            t=weibull_params_typical["eta"],
            beta=weibull_params_typical["beta"],
            eta=weibull_params_typical["eta"],
        )

        assert R == pytest.approx(math.exp(-1), rel=1e-6)  # 0.368

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_reliability_known_value(self):
        """
        Test reliability against known calculated value.

        R(t=10000, beta=2.5, eta=40000):
        R = exp(-(10000/40000)^2.5)
        R = exp(-(0.25)^2.5)
        R = exp(-0.03125)
        R = 0.9692...
        """
        R = weibull_reliability(t=10000, beta=2.5, eta=40000)

        expected = math.exp(-((10000 / 40000) ** 2.5))
        assert R == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_reliability_decreases_with_time(self, weibull_params_typical):
        """Test reliability decreases monotonically with time."""
        times = [0, 10000, 20000, 30000, 40000, 50000]
        reliabilities = [
            weibull_reliability(t, **weibull_params_typical)
            for t in times
        ]

        for i in range(len(reliabilities) - 1):
            assert reliabilities[i] > reliabilities[i + 1], (
                f"Reliability should decrease: R({times[i]})={reliabilities[i]:.4f} "
                f"> R({times[i+1]})={reliabilities[i+1]:.4f}"
            )

    @pytest.mark.unit
    def test_reliability_range(self, weibull_params_typical):
        """Test reliability is always between 0 and 1."""
        for t in [0, 1000, 10000, 50000, 100000, 200000]:
            R = weibull_reliability(t, **weibull_params_typical)
            assert 0 <= R <= 1

    @pytest.mark.unit
    def test_negative_time_raises(self, weibull_params_typical):
        """Test negative time raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            weibull_reliability(t=-100, **weibull_params_typical)

    @pytest.mark.unit
    def test_invalid_beta_raises(self):
        """Test invalid beta (<=0) raises ValueError."""
        with pytest.raises(ValueError, match="Beta must be > 0"):
            weibull_reliability(t=1000, beta=0, eta=40000)

        with pytest.raises(ValueError, match="Beta must be > 0"):
            weibull_reliability(t=1000, beta=-1.5, eta=40000)

    @pytest.mark.unit
    def test_invalid_eta_raises(self):
        """Test invalid eta (<=0) raises ValueError."""
        with pytest.raises(ValueError, match="Eta must be > 0"):
            weibull_reliability(t=1000, beta=2.5, eta=0)


# =============================================================================
# Test Class: Weibull Failure Rate (Hazard Function)
# =============================================================================


class TestWeibullFailureRate:
    """Tests for Weibull failure rate (hazard function) h(t)."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_failure_rate_formula(self):
        """
        Test failure rate against formula.

        h(t) = (beta/eta) * (t/eta)^(beta-1)
        At t=10000, beta=2.5, eta=40000:
        h = (2.5/40000) * (10000/40000)^1.5
        h = 6.25e-5 * 0.125
        h = 7.8125e-6 per hour
        """
        h = weibull_failure_rate(t=10000, beta=2.5, eta=40000)

        expected = (2.5 / 40000) * ((10000 / 40000) ** 1.5)
        assert h == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_failure_rate_at_zero_wearout(self):
        """Test failure rate at t=0 is zero for wear-out (beta>1)."""
        h = weibull_failure_rate(t=0, beta=2.5, eta=40000)
        assert h == 0.0

    @pytest.mark.unit
    def test_failure_rate_at_zero_infant_mortality(self):
        """Test failure rate at t=0 is infinite for infant mortality (beta<1)."""
        h = weibull_failure_rate(t=0, beta=0.7, eta=50000)
        assert h == float('inf')

    @pytest.mark.unit
    def test_failure_rate_constant_for_exponential(self):
        """Test failure rate is constant for beta=1 (exponential)."""
        # For exponential (beta=1), h(t) = 1/eta (constant)
        h1 = weibull_failure_rate(t=1000, beta=1.0, eta=30000)
        h2 = weibull_failure_rate(t=10000, beta=1.0, eta=30000)
        h3 = weibull_failure_rate(t=50000, beta=1.0, eta=30000)

        expected = 1.0 / 30000

        assert h1 == pytest.approx(expected, rel=1e-6)
        assert h2 == pytest.approx(expected, rel=1e-6)
        assert h3 == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_failure_rate_increases_wearout(self, weibull_params_wearout):
        """Test failure rate increases for wear-out failures (beta>1)."""
        times = [10000, 20000, 30000, 40000]
        rates = [weibull_failure_rate(t, **weibull_params_wearout) for t in times]

        for i in range(len(rates) - 1):
            assert rates[i + 1] > rates[i], "Failure rate should increase for wear-out"

    @pytest.mark.unit
    def test_failure_rate_decreases_infant_mortality(self, weibull_params_infant_mortality):
        """Test failure rate decreases for infant mortality (beta<1)."""
        times = [1000, 5000, 10000, 20000]
        rates = [weibull_failure_rate(t, **weibull_params_infant_mortality) for t in times]

        for i in range(len(rates) - 1):
            assert rates[i + 1] < rates[i], "Failure rate should decrease for infant mortality"


# =============================================================================
# Test Class: Weibull Mean Life (MTTF)
# =============================================================================


class TestWeibullMeanLife:
    """Tests for Weibull Mean Time To Failure (MTTF)."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_mttf_formula(self):
        """
        Test MTTF against formula.

        MTTF = eta * Gamma(1 + 1/beta)
        For beta=2.5, eta=40000:
        MTTF = 40000 * Gamma(1.4)
        MTTF = 40000 * 0.88726...
        MTTF = 35490.4...
        """
        mttf = weibull_mean_life(beta=2.5, eta=40000)

        expected = 40000 * math.gamma(1 + 1 / 2.5)
        assert mttf == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_mttf_equals_eta_for_exponential(self):
        """Test MTTF = eta for exponential distribution (beta=1)."""
        # Gamma(2) = 1! = 1, so MTTF = eta
        mttf = weibull_mean_life(beta=1.0, eta=30000)

        assert mttf == pytest.approx(30000, rel=1e-6)

    @pytest.mark.unit
    def test_mttf_less_than_eta_for_high_beta(self):
        """Test MTTF < eta for beta > 1 (typical wear-out)."""
        mttf = weibull_mean_life(beta=3.0, eta=40000)

        assert mttf < 40000


# =============================================================================
# Test Class: Remaining Useful Life (RUL)
# =============================================================================


class TestRemainingUsefulLife:
    """Tests for Remaining Useful Life calculation."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_rul_formula(self):
        """
        Test RUL calculation.

        For reliability threshold = 0.5 (median life):
        t_threshold = eta * (-ln(0.5))^(1/beta)
        RUL = t_threshold - current_hours
        """
        rul = remaining_useful_life(
            current_hours=10000,
            beta=2.5,
            eta=40000,
            reliability_threshold=0.5,
        )

        t_threshold = 40000 * ((-math.log(0.5)) ** (1 / 2.5))
        expected_rul = t_threshold - 10000

        assert rul == pytest.approx(expected_rul, rel=1e-6)

    @pytest.mark.unit
    def test_rul_at_zero_hours(self, weibull_params_typical):
        """Test RUL at zero operating hours equals full remaining life."""
        rul = remaining_useful_life(
            current_hours=0,
            beta=weibull_params_typical["beta"],
            eta=weibull_params_typical["eta"],
        )

        # Should be the median life
        t_median = weibull_params_typical["eta"] * ((-math.log(0.5)) ** (1 / weibull_params_typical["beta"]))
        assert rul == pytest.approx(t_median, rel=1e-6)

    @pytest.mark.unit
    def test_rul_is_zero_past_threshold(self, weibull_params_typical):
        """Test RUL is zero when already past threshold."""
        # Very long operating time - should have RUL = 0
        rul = remaining_useful_life(
            current_hours=200000,
            beta=weibull_params_typical["beta"],
            eta=weibull_params_typical["eta"],
        )

        assert rul == 0.0

    @pytest.mark.unit
    def test_rul_decreases_with_time(self, weibull_params_typical):
        """Test RUL decreases as operating hours increase."""
        times = [0, 10000, 20000, 30000]
        ruls = [
            remaining_useful_life(t, **weibull_params_typical)
            for t in times
        ]

        for i in range(len(ruls) - 1):
            assert ruls[i] > ruls[i + 1], "RUL should decrease with operating hours"


# =============================================================================
# Test Class: Failure Probability
# =============================================================================


class TestFailureProbability:
    """Tests for conditional failure probability calculation."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_failure_probability_formula(self):
        """
        Test failure probability formula.

        P(fail in [t1,t2] | survive to t1) = 1 - R(t2)/R(t1)
        """
        t_start = 10000
        t_end = 10720  # 30 days at 24 hr/day
        beta = 2.5
        eta = 40000

        prob = calculate_failure_probability(t_start, t_end, beta, eta)

        R_start = weibull_reliability(t_start, beta, eta)
        R_end = weibull_reliability(t_end, beta, eta)
        expected = 1 - (R_end / R_start)

        assert prob == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_failure_probability_range(self, weibull_params_typical):
        """Test failure probability is between 0 and 1."""
        prob = calculate_failure_probability(
            t_start=10000,
            t_end=20000,
            beta=weibull_params_typical["beta"],
            eta=weibull_params_typical["eta"],
        )

        assert 0 <= prob <= 1

    @pytest.mark.unit
    def test_failure_probability_increases_with_interval(self, weibull_params_typical):
        """Test failure probability increases with longer interval."""
        prob_short = calculate_failure_probability(
            t_start=10000,
            t_end=10500,
            **weibull_params_typical,
        )
        prob_long = calculate_failure_probability(
            t_start=10000,
            t_end=20000,
            **weibull_params_typical,
        )

        assert prob_long > prob_short

    @pytest.mark.unit
    def test_invalid_interval_raises(self, weibull_params_typical):
        """Test t_end <= t_start raises ValueError."""
        with pytest.raises(ValueError, match="t_end must be > t_start"):
            calculate_failure_probability(
                t_start=10000,
                t_end=5000,  # Before start
                **weibull_params_typical,
            )


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestBurnerInputValidation:
    """Tests for BurnerInput Pydantic model validation."""

    @pytest.mark.unit
    def test_valid_burner_input(self, burner_valid_input):
        """Test valid input passes validation."""
        assert burner_valid_input.burner_id == "BRN-001"
        assert burner_valid_input.operating_hours == 15000.0
        assert burner_valid_input.fuel_type == FuelType.NATURAL_GAS

    @pytest.mark.unit
    def test_weibull_params_validation(self):
        """Test Weibull parameters validation."""
        params = WeibullParameters(beta=2.5, eta=40000)

        assert params.beta == 2.5
        assert params.eta == 40000

    @pytest.mark.unit
    def test_weibull_beta_must_be_positive(self):
        """Test beta must be positive."""
        with pytest.raises(ValueError):
            WeibullParameters(beta=0, eta=40000)

    @pytest.mark.unit
    def test_flame_metrics_validation(self):
        """Test flame quality metrics validation."""
        metrics = FlameQualityMetrics(
            flame_temperature_c=1200,
            stability_index=0.95,
            o2_percent=3.5,
            co_ppm=50,
            nox_ppm=80,
        )

        assert metrics.stability_index == 0.95
        assert metrics.o2_percent == 3.5

    @pytest.mark.unit
    def test_stability_index_range(self):
        """Test stability index must be 0-1."""
        with pytest.raises(ValueError):
            FlameQualityMetrics(
                flame_temperature_c=1200,
                stability_index=1.5,  # Over 1
                o2_percent=3.5,
                co_ppm=50,
                nox_ppm=80,
            )

    @pytest.mark.unit
    def test_health_history_validation(self):
        """Test health history values must be 0-100."""
        # Should pass
        input_data = BurnerInput(
            burner_id="TEST-001",
            burner_model="TEST",
            fuel_type=FuelType.NATURAL_GAS,
            operating_hours=10000,
            design_life_hours=50000,
            weibull_params=WeibullParameters(beta=2.5, eta=40000),
            flame_metrics=FlameQualityMetrics(
                flame_temperature_c=1200,
                stability_index=0.9,
                o2_percent=3.5,
                co_ppm=50,
                nox_ppm=80,
            ),
            installation_date=date(2020, 1, 1),
            health_history=[95, 90, 85],  # Valid
        )
        assert len(input_data.health_history) == 3

        # Should fail
        with pytest.raises(ValueError):
            BurnerInput(
                burner_id="TEST-001",
                burner_model="TEST",
                fuel_type=FuelType.NATURAL_GAS,
                operating_hours=10000,
                design_life_hours=50000,
                weibull_params=WeibullParameters(beta=2.5, eta=40000),
                flame_metrics=FlameQualityMetrics(
                    flame_temperature_c=1200,
                    stability_index=0.9,
                    o2_percent=3.5,
                    co_ppm=50,
                    nox_ppm=80,
                ),
                installation_date=date(2020, 1, 1),
                health_history=[95, 110, 85],  # 110 is invalid
            )


# =============================================================================
# Test Class: Agent Execution
# =============================================================================


class TestBurnerAgentExecution:
    """Tests for burner agent run() method."""

    @pytest.mark.unit
    def test_healthy_burner_analysis(self, burner_agent, burner_valid_input):
        """Test analysis of healthy burner."""
        result = burner_agent.run(burner_valid_input)

        assert result.validation_status == "PASS"
        assert result.reliability > 0.8  # Should be high
        assert result.overall_health_score > 50
        assert result.maintenance_priority != MaintenancePriority.CRITICAL

    @pytest.mark.unit
    def test_degraded_burner_analysis(self, burner_agent, burner_degraded_input):
        """Test analysis of degraded burner."""
        result = burner_agent.run(burner_degraded_input)

        assert result.validation_status == "PASS"
        assert result.overall_health_score < 70
        assert result.maintenance_priority in [MaintenancePriority.MEDIUM, MaintenancePriority.HIGH, MaintenancePriority.CRITICAL]

    @pytest.mark.unit
    def test_critical_burner_analysis(self, burner_agent, burner_critical_input):
        """Test analysis of critical burner."""
        result = burner_agent.run(burner_critical_input)

        assert result.validation_status == "PASS"
        assert result.overall_health_score < 50
        assert result.should_replace is True

    @pytest.mark.unit
    def test_output_contains_all_metrics(self, burner_agent, burner_valid_input):
        """Test output contains all required metrics."""
        result = burner_agent.run(burner_valid_input)

        # Reliability metrics
        assert hasattr(result, "reliability")
        assert hasattr(result, "failure_rate")
        assert hasattr(result, "mttf_hours")
        assert hasattr(result, "remaining_useful_life_hours")
        assert hasattr(result, "failure_probability_30d")

        # Quality metrics
        assert hasattr(result, "flame_quality_score")
        assert hasattr(result, "combustion_efficiency")
        assert hasattr(result, "overall_health_score")
        assert hasattr(result, "degradation_rate")

        # Maintenance
        assert hasattr(result, "maintenance_priority")
        assert hasattr(result, "recommendations")
        assert hasattr(result, "should_replace")


# =============================================================================
# Test Class: Flame Quality Assessment
# =============================================================================


class TestFlameQualityAssessment:
    """Tests for flame quality scoring."""

    @pytest.mark.unit
    def test_high_quality_flame(self, burner_agent, burner_valid_input):
        """Test high quality flame gets high score."""
        result = burner_agent.run(burner_valid_input)

        # Good stability (0.95), low CO (50 ppm), optimal O2 (3.5%)
        assert result.flame_quality_score > 80

    @pytest.mark.unit
    def test_poor_quality_flame(self, burner_agent, burner_critical_input):
        """Test poor quality flame gets low score."""
        result = burner_agent.run(burner_critical_input)

        # Low stability (0.55), high CO (500 ppm), high O2 (7%)
        assert result.flame_quality_score < 60

    @pytest.mark.unit
    def test_flame_anomaly_detection(self, burner_agent, burner_degraded_input):
        """Test flame anomalies are detected."""
        result = burner_agent.run(burner_degraded_input)

        # High CO (200 ppm) and low stability should trigger anomalies
        assert len(result.flame_anomalies) > 0


# =============================================================================
# Test Class: Maintenance Recommendations
# =============================================================================


class TestMaintenanceRecommendations:
    """Tests for maintenance recommendation generation."""

    @pytest.mark.unit
    def test_recommendations_provided(self, burner_agent, burner_valid_input):
        """Test recommendations are provided."""
        result = burner_agent.run(burner_valid_input)

        assert isinstance(result.recommendations, list)

    @pytest.mark.unit
    def test_critical_burner_has_urgent_recommendations(self, burner_agent, burner_critical_input):
        """Test critical burner has urgent recommendations."""
        result = burner_agent.run(burner_critical_input)

        critical_recs = [r for r in result.recommendations if r.priority == MaintenancePriority.CRITICAL]
        assert len(critical_recs) > 0 or result.should_replace is True

    @pytest.mark.unit
    def test_recommendation_structure(self, burner_agent, burner_degraded_input):
        """Test recommendation has required fields."""
        result = burner_agent.run(burner_degraded_input)

        for rec in result.recommendations:
            assert hasattr(rec, "action")
            assert hasattr(rec, "priority")
            assert hasattr(rec, "reason")


# =============================================================================
# Test Class: Replacement Decision
# =============================================================================


class TestReplacementDecision:
    """Tests for replacement vs repair decision logic."""

    @pytest.mark.unit
    def test_no_replacement_for_healthy_burner(self, burner_agent, burner_valid_input):
        """Test healthy burner not recommended for replacement."""
        result = burner_agent.run(burner_valid_input)

        assert result.should_replace is False

    @pytest.mark.unit
    def test_replacement_for_critical_burner(self, burner_agent, burner_critical_input):
        """Test critical burner recommended for replacement."""
        result = burner_agent.run(burner_critical_input)

        assert result.should_replace is True
        assert result.replacement_reason is not None

    @pytest.mark.unit
    def test_repair_cost_ratio_affects_decision(self, burner_agent):
        """Test repair cost ratio affects replacement decision."""
        # High repair cost ratio should favor replacement
        input_high_cost = BurnerInput(
            burner_id="TEST-001",
            burner_model="ACME-5000",
            fuel_type=FuelType.NATURAL_GAS,
            operating_hours=30000,
            design_life_hours=50000,
            weibull_params=WeibullParameters(beta=2.5, eta=40000),
            flame_metrics=FlameQualityMetrics(
                flame_temperature_c=1100,
                stability_index=0.7,
                o2_percent=5.0,
                co_ppm=150,
                nox_ppm=100,
            ),
            installation_date=date(2018, 1, 1),
            repair_cost_ratio=0.8,  # High - favor replacement
        )

        result = burner_agent.run(input_high_cost)

        # With 80% repair cost ratio and degraded performance, replacement likely
        assert result.should_replace is True or result.maintenance_priority == MaintenancePriority.HIGH


# =============================================================================
# Test Class: Provenance Hash
# =============================================================================


class TestBurnerProvenanceHash:
    """Tests for provenance hash generation."""

    @pytest.mark.unit
    def test_provenance_hash_exists(self, burner_agent, burner_valid_input):
        """Test output includes provenance hash."""
        result = burner_agent.run(burner_valid_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_provenance_hash_valid_format(self, burner_agent, burner_valid_input):
        """Test provenance hash is valid SHA-256."""
        result = burner_agent.run(burner_valid_input)

        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())


# =============================================================================
# Test Class: Degradation Rate
# =============================================================================


class TestDegradationRate:
    """Tests for health degradation rate calculation."""

    @pytest.mark.unit
    def test_degradation_rate_from_history(self, burner_agent, burner_valid_input):
        """Test degradation rate calculated from health history."""
        result = burner_agent.run(burner_valid_input)

        # Health history: [95, 92, 89, 86, 83] - consistent decline
        assert result.degradation_rate > 0

    @pytest.mark.unit
    def test_no_degradation_rate_without_history(self, burner_agent):
        """Test degradation rate is 0 without sufficient history."""
        input_no_history = BurnerInput(
            burner_id="NEW-001",
            burner_model="ACME-5000",
            fuel_type=FuelType.NATURAL_GAS,
            operating_hours=1000,
            design_life_hours=50000,
            weibull_params=WeibullParameters(beta=2.5, eta=40000),
            flame_metrics=FlameQualityMetrics(
                flame_temperature_c=1200,
                stability_index=0.98,
                o2_percent=3.0,
                co_ppm=20,
                nox_ppm=60,
            ),
            installation_date=date.today() - timedelta(days=60),
            health_history=[],  # No history
        )

        result = burner_agent.run(input_no_history)

        assert result.degradation_rate == 0.0


# =============================================================================
# Test Class: Weibull Parameter Estimation
# =============================================================================


class TestWeibullParameterEstimation:
    """Tests for Weibull parameter estimation from failure data."""

    @pytest.mark.unit
    def test_estimate_parameters_from_failures(self):
        """Test parameter estimation from failure times."""
        failure_times = [15000, 18000, 22000, 25000, 30000, 32000, 35000]

        beta, eta = estimate_weibull_parameters(failure_times)

        # Parameters should be in reasonable range
        assert 0.5 <= beta <= 10.0
        assert eta > 0

    @pytest.mark.unit
    def test_insufficient_data_raises(self):
        """Test insufficient data raises error."""
        with pytest.raises(ValueError, match="at least 2 failure times"):
            estimate_weibull_parameters([10000])  # Only 1 point

    @pytest.mark.unit
    def test_percentile_life_calculation(self):
        """Test B10 life calculation (10% failure percentile)."""
        b10 = weibull_percentile_life(percentile=10, beta=2.5, eta=40000)

        # B10 should be less than eta
        assert 0 < b10 < 40000

    @pytest.mark.unit
    def test_median_life_equals_half_reliability(self):
        """Test median life (B50) corresponds to R=0.5."""
        b50 = weibull_percentile_life(percentile=50, beta=2.5, eta=40000)

        # At B50, reliability should be 0.5
        R_at_b50 = weibull_reliability(b50, beta=2.5, eta=40000)
        assert R_at_b50 == pytest.approx(0.5, rel=1e-4)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestBurnerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_new_burner_zero_hours(self, burner_agent):
        """Test analysis of brand new burner."""
        new_burner = BurnerInput(
            burner_id="NEW-001",
            burner_model="ACME-5000",
            fuel_type=FuelType.NATURAL_GAS,
            operating_hours=0,
            design_life_hours=50000,
            weibull_params=WeibullParameters(beta=2.5, eta=40000),
            flame_metrics=FlameQualityMetrics(
                flame_temperature_c=1200,
                stability_index=0.99,
                o2_percent=3.0,
                co_ppm=10,
                nox_ppm=50,
            ),
            installation_date=date.today(),
        )

        result = burner_agent.run(new_burner)

        assert result.reliability == 1.0
        assert result.overall_health_score > 90

    @pytest.mark.unit
    def test_very_old_burner(self, burner_agent):
        """Test analysis of very old burner past design life."""
        old_burner = BurnerInput(
            burner_id="OLD-001",
            burner_model="LEGACY-1000",
            fuel_type=FuelType.FUEL_OIL_2,
            operating_hours=80000,  # Past design life
            design_life_hours=50000,
            weibull_params=WeibullParameters(beta=3.0, eta=50000),
            flame_metrics=FlameQualityMetrics(
                flame_temperature_c=900,
                stability_index=0.5,
                o2_percent=8.0,
                co_ppm=600,
                nox_ppm=180,
            ),
            installation_date=date(2010, 1, 1),
        )

        result = burner_agent.run(old_burner)

        assert result.reliability < 0.5
        assert result.should_replace is True

    @pytest.mark.unit
    def test_all_fuel_types(self, burner_agent):
        """Test analysis works for all fuel types."""
        for fuel_type in FuelType:
            input_data = BurnerInput(
                burner_id=f"TEST-{fuel_type.value}",
                burner_model="UNIVERSAL-1000",
                fuel_type=fuel_type,
                operating_hours=10000,
                design_life_hours=50000,
                weibull_params=WeibullParameters(beta=2.5, eta=40000),
                flame_metrics=FlameQualityMetrics(
                    flame_temperature_c=1200,
                    stability_index=0.9,
                    o2_percent=3.5,
                    co_ppm=50,
                    nox_ppm=80,
                ),
                installation_date=date(2021, 1, 1),
            )

            result = burner_agent.run(input_data)

            assert result.validation_status == "PASS"


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestBurnerPerformance:
    """Performance tests for BurnerMaintenancePredictorAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_analysis_under_50ms(self, burner_agent, burner_valid_input, performance_timer):
        """Test single analysis completes in under 50ms."""
        performance_timer.start()
        result = burner_agent.run(burner_valid_input)
        performance_timer.stop()

        assert performance_timer.elapsed_ms < 50.0

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_analysis(self, burner_agent, performance_timer):
        """Test batch analysis throughput."""
        num_burners = 100
        inputs = [
            BurnerInput(
                burner_id=f"BRN-{i:03d}",
                burner_model="ACME-5000",
                fuel_type=FuelType.NATURAL_GAS,
                operating_hours=float(i * 500),
                design_life_hours=50000,
                weibull_params=WeibullParameters(beta=2.5, eta=40000),
                flame_metrics=FlameQualityMetrics(
                    flame_temperature_c=1200 - i,
                    stability_index=0.95 - i * 0.002,
                    o2_percent=3.5 + i * 0.01,
                    co_ppm=50 + i,
                    nox_ppm=80 + i * 0.5,
                ),
                installation_date=date(2020, 1, 1),
            )
            for i in range(num_burners)
        ]

        performance_timer.start()
        results = [burner_agent.run(inp) for inp in inputs]
        performance_timer.stop()

        assert len(results) == num_burners
        throughput = num_burners / (performance_timer.elapsed_ms / 1000)
        assert throughput >= 50, f"Throughput {throughput:.0f} rec/sec below target"
