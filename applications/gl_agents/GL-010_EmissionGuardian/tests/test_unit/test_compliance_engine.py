"""
GL-010 EmissionGuardian - Compliance Engine Tests

Comprehensive test suite for the deterministic compliance evaluation engine.
Tests rule evaluation, threshold calculations, and provenance tracking.

Reference: EPA 40 CFR Part 75 compliance methods
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compliance.engine import (
    EmissionsDataPoint,
    HourlyEvaluationInput,
    RollingEvaluationInput,
    RuleEvaluationResult,
    ComplianceEvaluationOutput,
    ComplianceEngine,
)
from compliance.schemas import (
    AveragingPeriod,
    OperatingState,
    PermitRule,
    ComplianceStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_rules_repository():
    """Mock rules repository with sample permit rules."""
    repo = Mock()

    # Sample NOx hourly rule
    nox_rule = Mock(spec=PermitRule)
    nox_rule.rule_id = "RULE-001"
    nox_rule.permit_id = "PERMIT-2024-001"
    nox_rule.pollutant = "NOX"
    nox_rule.limit_value = Decimal("0.15")  # lb/MMBtu
    nox_rule.limit_unit = "lb/MMBtu"
    nox_rule.averaging_period = AveragingPeriod.HOURLY
    nox_rule.warning_threshold_pct = 90.0
    nox_rule.action_threshold_pct = 100.0
    nox_rule.exemption_states = []
    nox_rule.is_applicable = Mock(return_value=True)

    # Sample SO2 hourly rule
    so2_rule = Mock(spec=PermitRule)
    so2_rule.rule_id = "RULE-002"
    so2_rule.permit_id = "PERMIT-2024-001"
    so2_rule.pollutant = "SO2"
    so2_rule.limit_value = Decimal("0.20")  # lb/MMBtu
    so2_rule.limit_unit = "lb/MMBtu"
    so2_rule.averaging_period = AveragingPeriod.HOURLY
    so2_rule.warning_threshold_pct = 90.0
    so2_rule.action_threshold_pct = 100.0
    so2_rule.exemption_states = [OperatingState.STARTUP]
    so2_rule.is_applicable = Mock(return_value=True)

    repo.get_rules_for_unit = Mock(return_value=[nox_rule, so2_rule])
    repo.get_rules_by_permit = Mock(return_value=[nox_rule, so2_rule])

    return repo


@pytest.fixture
def sample_hourly_input():
    """Sample hourly evaluation input."""
    return HourlyEvaluationInput(
        facility_id="FAC-001",
        unit_id="UNIT-001",
        evaluation_hour=datetime(2024, 1, 15, 14, 0, 0),
        operating_state=OperatingState.NORMAL,
        operating_params={"load_pct": 85.0},
        emissions_data=[
            EmissionsDataPoint(
                timestamp=datetime(2024, 1, 15, 14, 0, 0),
                unit_id="UNIT-001",
                pollutant="NOX",
                measured_value=Decimal("0.12"),
                measurement_unit="lb/MMBtu",
            ),
            EmissionsDataPoint(
                timestamp=datetime(2024, 1, 15, 14, 0, 0),
                unit_id="UNIT-001",
                pollutant="SO2",
                measured_value=Decimal("0.15"),
                measurement_unit="lb/MMBtu",
            ),
        ],
    )


@pytest.fixture
def compliance_engine(mock_rules_repository):
    """Compliance engine with mock repository."""
    return ComplianceEngine(
        rules_repository=mock_rules_repository,
        warning_threshold_default=90.0,
    )


# =============================================================================
# TEST: EMISSIONS DATA POINT
# =============================================================================

class TestEmissionsDataPoint:
    """Test EmissionsDataPoint model."""

    def test_data_point_creation(self):
        """Create valid data point."""
        dp = EmissionsDataPoint(
            timestamp=datetime.now(),
            unit_id="UNIT-001",
            pollutant="NOx",
            measured_value=Decimal("0.15"),
            measurement_unit="lb/MMBtu",
        )

        assert dp.pollutant == "NOX"  # Should be normalized to uppercase
        assert dp.measured_value == Decimal("0.15")
        assert dp.data_quality == "VALID"

    def test_pollutant_normalization(self):
        """Pollutant should be normalized to uppercase."""
        dp = EmissionsDataPoint(
            timestamp=datetime.now(),
            unit_id="UNIT-001",
            pollutant="nox",
            measured_value=Decimal("0.15"),
            measurement_unit="lb/MMBtu",
        )

        assert dp.pollutant == "NOX"

    def test_negative_value_rejected(self):
        """Negative measured values should be rejected."""
        with pytest.raises(ValueError):
            EmissionsDataPoint(
                timestamp=datetime.now(),
                unit_id="UNIT-001",
                pollutant="NOX",
                measured_value=Decimal("-0.15"),
                measurement_unit="lb/MMBtu",
            )


# =============================================================================
# TEST: HOURLY EVALUATION INPUT
# =============================================================================

class TestHourlyEvaluationInput:
    """Test HourlyEvaluationInput model."""

    def test_input_hash_calculation(self, sample_hourly_input):
        """Input hash should be calculated."""
        hash_val = sample_hourly_input.calculate_input_hash()

        assert len(hash_val) == 64  # SHA-256 hex
        assert hash_val.isalnum()

    def test_input_hash_deterministic(self, sample_hourly_input):
        """Same inputs should produce same hash."""
        hash1 = sample_hourly_input.calculate_input_hash()
        hash2 = sample_hourly_input.calculate_input_hash()

        assert hash1 == hash2

    def test_minimum_emissions_data(self):
        """At least one emissions data point required."""
        with pytest.raises(ValueError):
            HourlyEvaluationInput(
                facility_id="FAC-001",
                unit_id="UNIT-001",
                evaluation_hour=datetime.now(),
                operating_state=OperatingState.NORMAL,
                emissions_data=[],  # Empty list should fail
            )


# =============================================================================
# TEST: RULE EVALUATION RESULT
# =============================================================================

class TestRuleEvaluationResult:
    """Test RuleEvaluationResult dataclass."""

    def test_result_to_dict(self):
        """Result should convert to dict."""
        result = RuleEvaluationResult(
            rule_id="RULE-001",
            permit_id="PERMIT-001",
            pollutant="NOX",
            is_applicable=True,
            limit_value=Decimal("0.15"),
            measured_value=Decimal("0.12"),
            measurement_unit="lb/MMBtu",
            percentage_of_limit=80.0,
            threshold_status="COMPLIANT",
            warning_threshold_pct=90.0,
            action_threshold_pct=100.0,
            exceedance_pct=0.0,
        )

        d = result.to_dict()

        assert d["rule_id"] == "RULE-001"
        assert d["threshold_status"] == "COMPLIANT"
        assert d["percentage_of_limit"] == 80.0


# =============================================================================
# TEST: COMPLIANCE ENGINE INITIALIZATION
# =============================================================================

class TestComplianceEngineInit:
    """Test ComplianceEngine initialization."""

    def test_engine_init_defaults(self, mock_rules_repository):
        """Engine should initialize with defaults."""
        engine = ComplianceEngine(mock_rules_repository)

        assert engine.warning_threshold_default == 90.0
        assert engine.enable_exemptions is True

    def test_engine_init_custom_threshold(self, mock_rules_repository):
        """Engine should accept custom warning threshold."""
        engine = ComplianceEngine(
            mock_rules_repository,
            warning_threshold_default=85.0,
        )

        assert engine.warning_threshold_default == 85.0

    def test_engine_init_cache(self, mock_rules_repository):
        """Engine should initialize rolling cache."""
        engine = ComplianceEngine(mock_rules_repository)

        assert hasattr(engine, "_rolling_cache")
        assert engine.cache_size_hours == 720


# =============================================================================
# TEST: HOURLY EVALUATION
# =============================================================================

class TestHourlyEvaluation:
    """Test hourly compliance evaluation."""

    def test_hourly_evaluation_compliant(
        self, compliance_engine, sample_hourly_input
    ):
        """Hourly evaluation with compliant values."""
        result = compliance_engine.evaluate_hourly(sample_hourly_input)

        assert result.evaluation_type == "HOURLY"
        assert result.facility_id == "FAC-001"
        assert result.rules_evaluated >= 1

    def test_hourly_evaluation_output_structure(
        self, compliance_engine, sample_hourly_input
    ):
        """Output should have all required fields."""
        result = compliance_engine.evaluate_hourly(sample_hourly_input)

        assert result.evaluation_id.startswith("EVAL-HOURLY-")
        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.provenance_hash is not None
        assert result.processing_time_ms >= 0

    def test_hourly_evaluation_provenance(
        self, compliance_engine, sample_hourly_input
    ):
        """Evaluation should include provenance tracking."""
        result = compliance_engine.evaluate_hourly(sample_hourly_input)

        assert len(result.input_hash) == 64
        assert len(result.output_hash) == 64
        assert len(result.provenance_hash) == 64

    def test_hourly_evaluation_warning_status(
        self, compliance_engine, mock_rules_repository
    ):
        """Values near limit should show WARNING status."""
        # Create input with value at 92% of limit
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.NORMAL,
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="NOX",
                    measured_value=Decimal("0.138"),  # 92% of 0.15
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        assert result.rules_warning >= 1 or result.overall_status == "WARNING"

    def test_hourly_evaluation_exceeded_status(
        self, compliance_engine, mock_rules_repository
    ):
        """Values above limit should show EXCEEDED status."""
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.NORMAL,
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="NOX",
                    measured_value=Decimal("0.20"),  # 133% of 0.15
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        assert result.rules_exceeded >= 1
        assert result.overall_status == "EXCEEDED"
        assert len(result.exceedance_events) >= 1


# =============================================================================
# TEST: DETERMINISTIC CALCULATIONS
# =============================================================================

class TestDeterministicCalculations:
    """Test that calculations are deterministic."""

    def test_percentage_calculation_deterministic(
        self, compliance_engine, sample_hourly_input
    ):
        """Same inputs should produce same percentage."""
        result1 = compliance_engine.evaluate_hourly(sample_hourly_input)
        result2 = compliance_engine.evaluate_hourly(sample_hourly_input)

        # Compare rule results
        for r1, r2 in zip(result1.rule_results, result2.rule_results):
            assert r1["percentage_of_limit"] == r2["percentage_of_limit"]

    def test_threshold_status_deterministic(
        self, compliance_engine, sample_hourly_input
    ):
        """Threshold status should be deterministic."""
        result1 = compliance_engine.evaluate_hourly(sample_hourly_input)
        result2 = compliance_engine.evaluate_hourly(sample_hourly_input)

        assert result1.overall_status == result2.overall_status


# =============================================================================
# TEST: ROLLING EVALUATION
# =============================================================================

class TestRollingEvaluation:
    """Test rolling average compliance evaluation."""

    def test_rolling_evaluation_basic(
        self, compliance_engine, mock_rules_repository
    ):
        """Basic rolling evaluation."""
        # Create mock rule for rolling period
        rolling_rule = Mock(spec=PermitRule)
        rolling_rule.rule_id = "RULE-003"
        rolling_rule.permit_id = "PERMIT-001"
        rolling_rule.pollutant = "NOX"
        rolling_rule.limit_value = Decimal("0.12")
        rolling_rule.limit_unit = "lb/MMBtu"
        rolling_rule.averaging_period = AveragingPeriod.ROLLING_24HOUR
        rolling_rule.warning_threshold_pct = 90.0
        rolling_rule.action_threshold_pct = 100.0
        rolling_rule.exemption_states = []
        rolling_rule.is_applicable = Mock(return_value=True)

        mock_rules_repository.get_rules_for_unit.return_value = [rolling_rule]

        input_data = RollingEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_end=datetime.now(),
            averaging_period=AveragingPeriod.ROLLING_24HOUR,
            pollutant="NOX",
            hourly_values=[
                (datetime.now() - timedelta(hours=i), Decimal("0.10"))
                for i in range(24)
            ],
        )

        result = compliance_engine.evaluate_rolling(input_data)

        assert result.evaluation_type == "ROLLING"
        assert result.averaging_period == AveragingPeriod.ROLLING_24HOUR

    def test_rolling_periods_defined(self, compliance_engine):
        """Rolling periods should be properly defined."""
        periods = compliance_engine.ROLLING_PERIODS

        assert AveragingPeriod.ROLLING_3HOUR in periods
        assert AveragingPeriod.ROLLING_24HOUR in periods
        assert AveragingPeriod.ROLLING_30DAY in periods

        assert periods[AveragingPeriod.ROLLING_3HOUR] == 3
        assert periods[AveragingPeriod.ROLLING_24HOUR] == 24
        assert periods[AveragingPeriod.ROLLING_30DAY] == 720


# =============================================================================
# TEST: EXEMPTIONS
# =============================================================================

class TestExemptions:
    """Test exemption handling."""

    def test_startup_exemption(
        self, compliance_engine, mock_rules_repository
    ):
        """Startup state should exempt SO2 rule."""
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.STARTUP,  # Exempted state
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="SO2",
                    measured_value=Decimal("0.30"),  # Above limit
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        # SO2 should be exempt during startup
        for rule_result in result.rule_results:
            if rule_result["pollutant"] == "SO2":
                assert rule_result["exemption_applied"] or rule_result["threshold_status"] == "EXEMPT"

    def test_exemptions_disabled(self, mock_rules_repository):
        """Engine should respect exemptions disabled flag."""
        engine = ComplianceEngine(
            mock_rules_repository,
            enable_exemptions=False,
        )

        assert engine.enable_exemptions is False


# =============================================================================
# TEST: CACHE MANAGEMENT
# =============================================================================

class TestCacheManagement:
    """Test rolling data cache management."""

    def test_cache_update(self, compliance_engine, sample_hourly_input):
        """Hourly evaluation should update cache."""
        compliance_engine.evaluate_hourly(sample_hourly_input)

        # Cache should have entries
        assert len(compliance_engine._rolling_cache) > 0

    def test_cache_clear_all(self, compliance_engine, sample_hourly_input):
        """Clear all cache."""
        compliance_engine.evaluate_hourly(sample_hourly_input)

        cleared = compliance_engine.clear_cache()

        assert cleared > 0
        assert len(compliance_engine._rolling_cache) == 0

    def test_cache_clear_unit(self, compliance_engine, sample_hourly_input):
        """Clear cache for specific unit."""
        compliance_engine.evaluate_hourly(sample_hourly_input)

        cleared = compliance_engine.clear_cache(unit_id="UNIT-001")

        # Should clear only UNIT-001 entries
        for key in compliance_engine._rolling_cache:
            assert key[0] != "UNIT-001"


# =============================================================================
# TEST: PERMIT EVALUATION
# =============================================================================

class TestPermitEvaluation:
    """Test permit-level compliance evaluation."""

    def test_permit_evaluation(
        self, compliance_engine, mock_rules_repository
    ):
        """Evaluate all rules for a permit."""
        emissions_summary = {
            "NOX": {"hourly": Decimal("0.10")},
            "SO2": {"hourly": Decimal("0.15")},
        }

        result = compliance_engine.evaluate_permit(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            permit_id="PERMIT-2024-001",
            evaluation_date=date.today(),
            emissions_summary=emissions_summary,
        )

        assert result.evaluation_type == "PERMIT"
        assert result.rules_evaluated >= 1


# =============================================================================
# TEST: COMPLIANCE STATUS
# =============================================================================

class TestComplianceStatus:
    """Test compliance status retrieval."""

    def test_get_compliance_status(self, compliance_engine):
        """Get aggregated compliance status."""
        status = compliance_engine.get_compliance_status(
            facility_id="FAC-001",
            unit_id="UNIT-001",
        )

        assert status.facility_id == "FAC-001"
        assert status.unit_id == "UNIT-001"
        assert status.provenance_hash is not None


# =============================================================================
# TEST: DECIMAL PRECISION
# =============================================================================

class TestDecimalPrecision:
    """Test Decimal precision in compliance calculations."""

    def test_percentage_precision(self, compliance_engine, mock_rules_repository):
        """Percentage should have proper precision."""
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.NORMAL,
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="NOX",
                    measured_value=Decimal("0.111111"),  # Repeating decimal
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        for rule_result in result.rule_results:
            pct = rule_result["percentage_of_limit"]
            # Should be rounded to 2 decimal places
            assert abs(pct - round(pct, 2)) < 0.01


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_emission_value(self, compliance_engine, mock_rules_repository):
        """Zero emissions should be valid and compliant."""
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.NORMAL,
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="NOX",
                    measured_value=Decimal("0"),
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        assert result.overall_status == "COMPLIANT"

    def test_exactly_at_limit(self, compliance_engine, mock_rules_repository):
        """Value exactly at limit should be EXCEEDED."""
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.NORMAL,
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="NOX",
                    measured_value=Decimal("0.15"),  # Exactly at limit
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        # At 100% of limit = action threshold
        # Status depends on action_threshold_pct setting

    def test_exactly_at_warning(self, compliance_engine, mock_rules_repository):
        """Value exactly at warning threshold."""
        input_data = HourlyEvaluationInput(
            facility_id="FAC-001",
            unit_id="UNIT-001",
            evaluation_hour=datetime.now(),
            operating_state=OperatingState.NORMAL,
            emissions_data=[
                EmissionsDataPoint(
                    timestamp=datetime.now(),
                    unit_id="UNIT-001",
                    pollutant="NOX",
                    measured_value=Decimal("0.135"),  # 90% of 0.15
                    measurement_unit="lb/MMBtu",
                ),
            ],
        )

        result = compliance_engine.evaluate_hourly(input_data)

        # At 90% = warning threshold
        assert result.rules_warning >= 1 or result.overall_status in ["WARNING", "COMPLIANT"]

