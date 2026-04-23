# -*- coding: utf-8 -*-
"""
Unit tests for FailurePredictor.

Tests Weibull survival analysis, uncertainty quantification, and risk assessment.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
import math
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diagnostics.failure_predictor import (
    FailurePredictor,
    PredictorConfig,
    TrapHistory,
    TrapType,
    RiskLevel,
    FailureMode,
)


class TestFailurePredictor:
    """Tests for FailurePredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create default predictor."""
        return FailurePredictor()

    @pytest.fixture
    def young_trap(self):
        """Create history for young trap."""
        return TrapHistory(
            trap_id="ST-YOUNG",
            trap_type=TrapType.THERMODYNAMIC,
            install_date=datetime.now(timezone.utc),
            age_years=1.0,
            pressure_bar_g=10.0,
        )

    @pytest.fixture
    def old_trap(self):
        """Create history for old trap."""
        return TrapHistory(
            trap_id="ST-OLD",
            trap_type=TrapType.THERMODYNAMIC,
            install_date=datetime.now(timezone.utc),
            age_years=8.0,
            previous_failures=2,
            pressure_bar_g=10.0,
        )

    @pytest.fixture
    def stressed_trap(self):
        """Create history for trap under stress."""
        return TrapHistory(
            trap_id="ST-STRESS",
            trap_type=TrapType.THERMODYNAMIC,
            install_date=datetime.now(timezone.utc),
            age_years=3.0,
            pressure_bar_g=20.0,  # High pressure
            has_water_hammer=True,
            is_cycling=True,
            has_dirty_steam=True,
        )

    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor is not None
        assert predictor.config is not None

    def test_predict_young_trap(self, predictor, young_trap):
        """Test prediction for young trap."""
        prediction = predictor.predict_failure(young_trap)

        assert prediction is not None
        assert prediction.trap_id == "ST-YOUNG"
        assert prediction.failure_probability.point_estimate < 0.5
        assert prediction.risk_assessment.risk_level in (RiskLevel.VERY_LOW, RiskLevel.LOW)

    def test_predict_old_trap(self, predictor, old_trap):
        """Test prediction for old trap."""
        prediction = predictor.predict_failure(old_trap)

        assert prediction is not None
        assert prediction.failure_probability.point_estimate > 0.2
        assert prediction.risk_assessment.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.VERY_HIGH)

    def test_predict_stressed_trap(self, predictor, stressed_trap):
        """Test prediction for stressed trap."""
        prediction = predictor.predict_failure(stressed_trap)

        # Stressed trap should have higher failure probability than normal
        normal_trap = TrapHistory(
            trap_id="ST-NORMAL",
            trap_type=TrapType.THERMODYNAMIC,
            install_date=datetime.now(timezone.utc),
            age_years=3.0,
            pressure_bar_g=10.0,
        )
        normal_pred = predictor.predict_failure(normal_trap)

        assert prediction.failure_probability.point_estimate > normal_pred.failure_probability.point_estimate

    def test_probability_bounds(self, predictor, young_trap):
        """Test probability is within valid range."""
        prediction = predictor.predict_failure(young_trap)

        assert 0.0 <= prediction.failure_probability.point_estimate <= 1.0
        assert 0.0 <= prediction.failure_probability.lower_bound <= 1.0
        assert 0.0 <= prediction.failure_probability.upper_bound <= 1.0

    def test_uncertainty_interval(self, predictor, young_trap):
        """Test uncertainty interval is valid."""
        prediction = predictor.predict_failure(young_trap)

        prob = prediction.failure_probability
        assert prob.lower_bound <= prob.point_estimate
        assert prob.point_estimate <= prob.upper_bound

    def test_remaining_life_estimate(self, predictor, young_trap):
        """Test remaining life estimate."""
        prediction = predictor.predict_failure(young_trap)

        rul = prediction.expected_remaining_life_days
        assert rul.point_estimate >= 0
        assert rul.lower_bound <= rul.point_estimate
        assert rul.point_estimate <= rul.upper_bound

    def test_hazard_rate_calculation(self, predictor, young_trap, old_trap):
        """Test hazard rate calculation."""
        young_pred = predictor.predict_failure(young_trap)
        old_pred = predictor.predict_failure(old_trap)

        # For Weibull with beta > 1, hazard rate increases with age
        assert old_pred.hazard_rate > young_pred.hazard_rate

    def test_reliability_calculation(self, predictor, young_trap, old_trap):
        """Test reliability calculation."""
        young_pred = predictor.predict_failure(young_trap)
        old_pred = predictor.predict_failure(old_trap)

        assert young_pred.current_reliability > old_pred.current_reliability
        assert 0.0 <= young_pred.current_reliability <= 1.0

    def test_risk_assessment(self, predictor, old_trap):
        """Test risk assessment is complete."""
        prediction = predictor.predict_failure(old_trap)
        risk = prediction.risk_assessment

        assert risk.risk_level is not None
        assert 0 <= risk.risk_score <= 100
        assert risk.dominant_failure_mode is not None
        assert len(risk.failure_mode_probabilities) > 0
        assert len(risk.recommended_actions) > 0

    def test_contributing_factors(self, predictor, stressed_trap):
        """Test contributing factors identification."""
        prediction = predictor.predict_failure(stressed_trap)

        factors = prediction.risk_assessment.contributing_factors
        assert len(factors) > 0
        # Should identify some stress factors
        factor_str = " ".join(factors).lower()
        assert "pressure" in factor_str or "water hammer" in factor_str or "cycling" in factor_str

    def test_deterministic_prediction(self, predictor, young_trap):
        """Test that same input produces same output."""
        pred1 = predictor.predict_failure(young_trap)
        pred2 = predictor.predict_failure(young_trap)

        assert pred1.failure_probability.point_estimate == pred2.failure_probability.point_estimate
        assert pred1.provenance_hash == pred2.provenance_hash

    def test_fleet_prediction(self, predictor, young_trap, old_trap, stressed_trap):
        """Test fleet-wide prediction."""
        histories = [young_trap, old_trap, stressed_trap]
        predictions = predictor.predict_fleet(histories)

        assert len(predictions) == 3
        # Should be sorted by failure probability (highest first)
        for i in range(len(predictions) - 1):
            assert predictions[i].failure_probability.point_estimate >= \
                   predictions[i+1].failure_probability.point_estimate

    def test_risk_report_generation(self, predictor, young_trap, old_trap):
        """Test risk report generation."""
        predictions = predictor.predict_fleet([young_trap, old_trap])
        report = predictor.generate_risk_report(predictions)

        assert "RISK ASSESSMENT REPORT" in report
        assert "FLEET RISK SUMMARY" in report

    def test_prediction_to_dict(self, predictor, young_trap):
        """Test prediction serialization."""
        prediction = predictor.predict_failure(young_trap)
        pred_dict = prediction.to_dict()

        assert "trap_id" in pred_dict
        assert "failure_probability" in pred_dict
        assert "risk_assessment" in pred_dict

    def test_trap_type_affects_prediction(self, predictor):
        """Test that trap type affects prediction."""
        thermodynamic = TrapHistory(
            trap_id="TD",
            trap_type=TrapType.THERMODYNAMIC,
            install_date=datetime.now(timezone.utc),
            age_years=5.0,
        )

        orifice = TrapHistory(
            trap_id="OR",
            trap_type=TrapType.ORIFICE,
            install_date=datetime.now(timezone.utc),
            age_years=5.0,
        )

        td_pred = predictor.predict_failure(thermodynamic)
        or_pred = predictor.predict_failure(orifice)

        # Different trap types should have different predictions
        assert td_pred.failure_probability.point_estimate != or_pred.failure_probability.point_estimate


class TestWeibullFunctions:
    """Tests for internal Weibull calculations."""

    @pytest.fixture
    def predictor(self):
        return FailurePredictor()

    def test_weibull_reliability_at_zero(self, predictor):
        """Test reliability at t=0 is 1."""
        r = predictor._weibull_reliability(0, 2.0, 5.0)
        assert r == 1.0

    def test_weibull_reliability_at_eta(self, predictor):
        """Test reliability at t=eta is e^(-1)."""
        r = predictor._weibull_reliability(5.0, 1.0, 5.0)  # beta=1 is exponential
        expected = math.exp(-1)
        assert abs(r - expected) < 0.001

    def test_weibull_quantile_inverse(self, predictor):
        """Test quantile is inverse of CDF."""
        beta, eta = 2.0, 5.0
        p = 0.5

        t = predictor._weibull_quantile(p, beta, eta)
        # F(t) = 1 - R(t) should equal p
        f_t = 1 - predictor._weibull_reliability(t, beta, eta)

        assert abs(f_t - p) < 0.001
