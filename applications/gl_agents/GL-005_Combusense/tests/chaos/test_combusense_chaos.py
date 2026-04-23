"""
GL-005 Combusense - Chaos Engineering Test Suite

This module contains pytest-based chaos engineering tests specific to
the Combusense Emissions Analytics Agent.

Test Categories:
- CEMS fault scenarios
- Regulatory reporting failures
- Data quality degradation
- Correlation engine failures
- Predictive model failures

All tests are CI-safe (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import pytest
import logging
import sys
import os
from datetime import datetime, timezone

gl001_chaos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
if gl001_chaos_path not in sys.path:
    sys.path.insert(0, gl001_chaos_path)

from chaos_runner import ChaosRunner, ChaosExperiment, ChaosSeverity
from steady_state import SteadyStateValidator

from .combusense_chaos import (
    CombusenseChaosConfig,
    CEMSFaultInjector,
    RegulatoryReportingFaultInjector,
    DataQualityFaultInjector,
    CorrelationEngineFaultInjector,
    PredictiveModelFaultInjector,
    CEMSStatus,
    DataQualityLevel,
    create_combusense_hypothesis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cems_injector():
    return CEMSFaultInjector()


@pytest.fixture
def reporting_injector():
    return RegulatoryReportingFaultInjector()


@pytest.fixture
def quality_injector():
    return DataQualityFaultInjector()


@pytest.fixture
def correlation_injector():
    return CorrelationEngineFaultInjector()


@pytest.fixture
def prediction_injector():
    return PredictiveModelFaultInjector()


# =============================================================================
# CEMS Fault Tests
# =============================================================================

class TestCEMSFaults:
    """Test suite for CEMS fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_analyzer_failure(self, cems_injector):
        """Test CEMS analyzer failure."""
        await cems_injector.inject({
            "fault_type": "analyzer_failure",
            "analyzer": "nox_analyzer",
        })

        status = cems_injector.get_analyzer_status("nox_analyzer")
        assert status == CEMSStatus.FAULT

        availability = cems_injector.get_data_availability("nox_analyzer")
        assert availability == 0.0

        reading = cems_injector.get_reading("nox_analyzer")
        assert reading["value"] is None
        assert reading["quality"] == "invalid"

        await cems_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calibration_drift(self, cems_injector):
        """Test CEMS calibration drift."""
        await cems_injector.inject({
            "fault_type": "calibration_drift",
            "analyzer": "so2_analyzer",
            "drift_percent": 10,
        })

        status = cems_injector.get_analyzer_status("so2_analyzer")
        assert status == CEMSStatus.CALIBRATING

        reading = cems_injector.get_reading("so2_analyzer")
        assert reading["value"] is None
        assert reading["quality"] == "calibrating"

        await cems_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_communication_failure(self, cems_injector):
        """Test CEMS communication failure (all analyzers offline)."""
        await cems_injector.inject({
            "fault_type": "communication_failure",
        })

        # All analyzers should be offline
        for analyzer in ["nox_analyzer", "so2_analyzer", "co_analyzer"]:
            status = cems_injector.get_analyzer_status(analyzer)
            assert status == CEMSStatus.OFFLINE

        await cems_injector.rollback()


# =============================================================================
# Regulatory Reporting Fault Tests
# =============================================================================

class TestRegulatoryReportingFaults:
    """Test suite for regulatory reporting fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_submission_failure(self, reporting_injector):
        """Test report submission failure."""
        await reporting_injector.inject({
            "fault_type": "submission_failure",
            "failure_reason": "network_timeout",
        })

        result = await reporting_injector.submit_report({
            "report_type": "quarterly_emissions",
            "period": "2024-Q1",
        })

        assert result["status"] == "failed"
        assert "network_timeout" in result.get("error", "")

        await reporting_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_validation_error(self, reporting_injector):
        """Test report validation error."""
        await reporting_injector.inject({
            "fault_type": "validation_error",
        })

        result = await reporting_injector.submit_report({
            "report_type": "monthly_emissions",
        })

        assert result["status"] == "rejected"
        assert "validation_failed" in result.get("error", "")
        assert "details" in result

        await reporting_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_credential_expiration(self, reporting_injector):
        """Test credential expiration."""
        await reporting_injector.inject({
            "fault_type": "credential_expired",
        })

        result = await reporting_injector.submit_report({})

        assert result["status"] == "unauthorized"

        await reporting_injector.rollback()


# =============================================================================
# Data Quality Fault Tests
# =============================================================================

class TestDataQualityFaults:
    """Test suite for data quality fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_missing_data(self, quality_injector):
        """Test missing data injection."""
        await quality_injector.inject({
            "fault_type": "missing_data",
            "missing_percent": 50,
        })

        # Process multiple data points
        results = []
        for i in range(100):
            result = quality_injector.process_data_point(
                value=30.0 + i,
                timestamp=datetime.now(timezone.utc)
            )
            results.append(result)

        missing_count = sum(
            1 for r in results if r["quality"] == DataQualityLevel.MISSING.value
        )

        # Should have roughly 50% missing (with statistical variance)
        assert 30 < missing_count < 70

        await quality_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_outlier_injection(self, quality_injector):
        """Test outlier data injection."""
        await quality_injector.inject({
            "fault_type": "outliers",
            "outlier_percent": 20,
        })

        results = []
        for i in range(100):
            result = quality_injector.process_data_point(
                value=30.0,
                timestamp=datetime.now(timezone.utc)
            )
            results.append(result)

        outliers = [r for r in results if r.get("flag") == "outlier_detected"]
        assert len(outliers) > 0

        # Outliers should have extreme values
        for outlier in outliers:
            assert abs(outlier["value"]) > 100

        await quality_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_data_quality_score(self, quality_injector):
        """Test data quality score calculation."""
        # Without faults
        data_points = [
            {"quality": DataQualityLevel.VALID.value},
            {"quality": DataQualityLevel.VALID.value},
            {"quality": DataQualityLevel.VALID.value},
        ]
        score = quality_injector.get_quality_score(data_points)
        assert score == 1.0

        # With some invalid data
        data_points.append({"quality": DataQualityLevel.MISSING.value})
        score = quality_injector.get_quality_score(data_points)
        assert score == 0.75


# =============================================================================
# Correlation Engine Fault Tests
# =============================================================================

class TestCorrelationEngineFaults:
    """Test suite for correlation engine fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calculation_timeout(self, correlation_injector):
        """Test correlation calculation timeout."""
        await correlation_injector.inject({
            "fault_type": "calculation_timeout",
            "timeout_ms": 100,  # Short for testing
        })

        series_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        series_b = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = await correlation_injector.calculate_correlation(series_a, series_b)

        assert result["status"] == "timeout"

        await correlation_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_normal_correlation(self, correlation_injector):
        """Test normal correlation calculation (no fault)."""
        series_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        series_b = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = await correlation_injector.calculate_correlation(series_a, series_b)

        assert result["status"] == "success"
        assert abs(result["correlation"] - 1.0) < 0.01  # Perfect correlation

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_memory_overflow(self, correlation_injector):
        """Test memory overflow error."""
        await correlation_injector.inject({
            "fault_type": "memory_overflow",
        })

        result = await correlation_injector.calculate_correlation([1, 2, 3], [4, 5, 6])

        assert result["status"] == "error"
        assert "memory_overflow" in result.get("error", "")

        await correlation_injector.rollback()


# =============================================================================
# Predictive Model Fault Tests
# =============================================================================

class TestPredictiveModelFaults:
    """Test suite for predictive model fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_model_timeout(self, prediction_injector):
        """Test model prediction timeout."""
        await prediction_injector.inject({
            "fault_type": "model_timeout",
            "timeout_ms": 100,  # Short for testing
        })

        result = await prediction_injector.predict(
            features={"temperature": 25.0, "load": 80.0},
            horizon_hours=24
        )

        assert result["status"] == "timeout"

        await prediction_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_poor_accuracy(self, prediction_injector):
        """Test degraded prediction accuracy."""
        await prediction_injector.inject({
            "fault_type": "poor_accuracy",
            "error_margin": 40,
        })

        result = await prediction_injector.predict(
            features={"temperature": 25.0},
            horizon_hours=12
        )

        assert result["status"] == "degraded"
        assert result["accuracy"] < 70
        assert "accuracy_degraded" in result.get("warning", "")

        await prediction_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_missing_feature(self, prediction_injector):
        """Test missing feature error."""
        await prediction_injector.inject({
            "fault_type": "feature_unavailable",
            "missing_feature": "humidity",
        })

        result = await prediction_injector.predict(
            features={"temperature": 25.0},
            horizon_hours=6
        )

        assert result["status"] == "error"
        assert "missing_feature" in result.get("error", "")

        await prediction_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_out_of_bounds_predictions(self, prediction_injector):
        """Test out-of-bounds predictions."""
        await prediction_injector.inject({
            "fault_type": "out_of_bounds",
        })

        result = await prediction_injector.predict(
            features={},
            horizon_hours=4
        )

        assert result["status"] == "warning"
        assert "out_of_valid_range" in result.get("warning", "")

        # Check for extreme values
        predictions = result.get("predictions", [])
        has_extreme = any(
            p["value"] < 0 or p["value"] > 500
            for p in predictions
        )
        assert has_extreme

        await prediction_injector.rollback()


# =============================================================================
# Steady State Hypothesis Tests
# =============================================================================

class TestCombusenseSteadyState:
    """Test suite for Combusense steady state validation."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_combusense_hypothesis(self):
        """Test Combusense-specific steady state hypothesis."""
        hypothesis = create_combusense_hypothesis()

        assert hypothesis.name == "Combusense Emissions Analytics Health"
        assert len(hypothesis.metrics) >= 3

        validator = SteadyStateValidator()
        result = await validator.validate(hypothesis)

        assert result.hypothesis_name == hypothesis.name


# =============================================================================
# Integration Tests
# =============================================================================

class TestCombusenseChaosIntegration:
    """Integration tests for Combusense chaos scenarios."""

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cems_to_reporting_cascade(
        self,
        cems_injector,
        quality_injector,
        reporting_injector
    ):
        """Test cascading failure from CEMS to reporting."""
        # Stage 1: CEMS analyzer failure
        await cems_injector.inject({
            "fault_type": "analyzer_failure",
            "analyzer": "nox_analyzer",
        })

        reading = cems_injector.get_reading("nox_analyzer")
        assert reading["value"] is None

        # Stage 2: Data quality degradation
        await quality_injector.inject({
            "fault_type": "missing_data",
            "missing_percent": 30,
        })

        # Stage 3: Report submission with degraded data
        await reporting_injector.inject({
            "fault_type": "validation_error",
        })

        result = await reporting_injector.submit_report({
            "report_type": "quarterly",
        })
        assert result["status"] == "rejected"

        # Cleanup
        await reporting_injector.rollback()
        await quality_injector.rollback()
        await cems_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_prediction_with_data_quality_issues(
        self,
        quality_injector,
        correlation_injector,
        prediction_injector
    ):
        """Test prediction accuracy with data quality issues."""
        # Inject data quality issues
        await quality_injector.inject({
            "fault_type": "outliers",
            "outlier_percent": 15,
        })

        # Correlation might fail with bad data
        await correlation_injector.inject({
            "fault_type": "invalid_result",
        })

        correlation_result = await correlation_injector.calculate_correlation(
            [1, 2, 3], [4, 5, 6]
        )
        assert correlation_result["status"] == "warning"

        # Predictions should be degraded
        await prediction_injector.inject({
            "fault_type": "poor_accuracy",
            "error_margin": 25,
        })

        prediction_result = await prediction_injector.predict({}, 12)
        assert prediction_result["status"] == "degraded"

        # Cleanup
        await prediction_injector.rollback()
        await correlation_injector.rollback()
        await quality_injector.rollback()


def pytest_configure(config):
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")
    config.addinivalue_line("markers", "integration: Integration tests")
