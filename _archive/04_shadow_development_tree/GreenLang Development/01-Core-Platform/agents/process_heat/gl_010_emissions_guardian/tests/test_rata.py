# -*- coding: utf-8 -*-
"""
GL-010 RATA Automation Tests
============================

Unit tests for EPA RATA (Relative Accuracy Test Audit) automation.
Tests 40 CFR Part 75 compliance calculations and scheduling.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta, date
from typing import List

from greenlang.agents.process_heat.gl_010_emissions_guardian.rata_automation import (
    RATAAutomation,
    RATARunData,
    RATATestInput,
    RATAResult,
    CGAResult,
    CalibrationDriftResult,
    RATASchedule,
    RATAConstants,
    RATAFrequency,
    CEMSPollutant,
    TestStatus,
    BiasAdjustmentStatus,
)


class TestRATAConstants:
    """Tests for RATA regulatory constants."""

    def test_ra_thresholds(self):
        """Test RA threshold values."""
        assert RATAConstants.RA_THRESHOLD_PERCENT == 10.0
        assert RATAConstants.RA_THRESHOLD_ALTERNATE == 20.0
        assert RATAConstants.RA_THRESHOLD_DILUENT == 1.0

    def test_bias_threshold(self):
        """Test bias threshold."""
        assert RATAConstants.BIAS_THRESHOLD_PERCENT == 5.0

    def test_cga_threshold(self):
        """Test CGA tolerance."""
        assert RATAConstants.CGA_TOLERANCE_PERCENT == 5.0

    def test_run_requirements(self):
        """Test run requirements."""
        assert RATAConstants.MIN_RATA_RUNS == 9
        assert RATAConstants.MAX_INVALID_RUNS == 3
        assert RATAConstants.MIN_RUN_DURATION_MINUTES == 21

    def test_epa_method_ranges(self):
        """Test EPA method ranges."""
        assert RATAConstants.EPA_METHOD_7E_RANGE == (0, 5000)
        assert RATAConstants.EPA_METHOD_3A_O2_RANGE == (0, 21)


class TestRATARunData:
    """Tests for RATA run data model."""

    def test_valid_run(self):
        """Test valid RATA run data."""
        run = RATARunData(
            run_number=1,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            reference_value=100.0,
            reference_unit="ppm",
            cems_value=98.0,
            load_percent=85.0,
        )
        assert run.run_number == 1
        assert run.is_valid is True
        assert run.duration_minutes >= 21

    def test_difference_calculation(self):
        """Test difference calculation."""
        run = RATARunData(
            run_number=1,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            reference_value=100.0,
            reference_unit="ppm",
            cems_value=95.0,
            load_percent=85.0,
        )
        assert run.difference == 5.0  # reference - cems

    def test_duration_minutes(self):
        """Test duration calculation."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(minutes=45)

        run = RATARunData(
            run_number=1,
            start_time=start,
            end_time=end,
            reference_value=100.0,
            reference_unit="ppm",
            cems_value=98.0,
            load_percent=85.0,
        )
        assert abs(run.duration_minutes - 45.0) < 0.01

    def test_minimum_duration_validation(self):
        """Test minimum duration validation."""
        with pytest.raises(ValueError):
            RATARunData(
                run_number=1,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc) + timedelta(minutes=15),  # Too short
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=98.0,
                load_percent=85.0,
            )

    def test_invalid_run(self):
        """Test invalid run."""
        run = RATARunData(
            run_number=1,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            reference_value=100.0,
            reference_unit="ppm",
            cems_value=98.0,
            load_percent=85.0,
            is_valid=False,
            invalidation_reason="Equipment malfunction",
        )
        assert run.is_valid is False
        assert run.invalidation_reason == "Equipment malfunction"


class TestRATATestInput:
    """Tests for RATA test input model."""

    @pytest.fixture
    def valid_runs(self) -> List[RATARunData]:
        """Create valid RATA runs."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=100.0 + i,
                reference_unit="ppm",
                cems_value=98.0 + i,
                load_percent=85.0,
            ))
        return runs

    def test_valid_input(self, valid_runs):
        """Test valid RATA test input."""
        test_input = RATATestInput(
            unit_id="UNIT-001",
            pollutant=CEMSPollutant.NOX,
            test_date=date.today(),
            reference_method="7E",
            analyzer_span=500.0,
            runs=valid_runs,
        )
        assert test_input.unit_id == "UNIT-001"
        assert len(test_input.runs) == 9

    def test_minimum_runs_validation(self):
        """Test minimum runs validation."""
        runs = [
            RATARunData(
                run_number=i,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc) + timedelta(minutes=30),
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=98.0,
                load_percent=85.0,
            )
            for i in range(1, 6)  # Only 5 runs
        ]

        with pytest.raises(ValueError):
            RATATestInput(
                unit_id="UNIT-001",
                pollutant=CEMSPollutant.NOX,
                test_date=date.today(),
                reference_method="7E",
                analyzer_span=500.0,
                runs=runs,
            )


class TestRATAAutomation:
    """Tests for RATA automation engine."""

    @pytest.fixture
    def rata_engine(self):
        """Create RATA automation engine."""
        return RATAAutomation(unit_id="UNIT-001")

    @pytest.fixture
    def passing_test_data(self) -> RATATestInput:
        """Create RATA test data that should pass."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            # Small differences for passing test
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=98.0 + (i * 0.2),  # Small variation
                load_percent=85.0,
            ))

        return RATATestInput(
            unit_id="UNIT-001",
            pollutant=CEMSPollutant.NOX,
            test_date=date.today(),
            reference_method="7E",
            analyzer_span=500.0,
            runs=runs,
        )

    @pytest.fixture
    def failing_test_data(self) -> RATATestInput:
        """Create RATA test data that should fail."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            # Large differences for failing test
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=80.0 + (i * 2),  # Large variation
                load_percent=85.0,
            ))

        return RATATestInput(
            unit_id="UNIT-001",
            pollutant=CEMSPollutant.NOX,
            test_date=date.today(),
            reference_method="7E",
            analyzer_span=500.0,
            runs=runs,
        )

    def test_initialization(self, rata_engine):
        """Test RATA engine initialization."""
        assert rata_engine.unit_id == "UNIT-001"
        assert len(rata_engine.pollutants) == 5
        assert rata_engine.current_frequency == RATAFrequency.QUARTERLY

    def test_custom_pollutants(self):
        """Test initialization with custom pollutants."""
        engine = RATAAutomation(
            unit_id="UNIT-001",
            pollutants=[CEMSPollutant.NOX, CEMSPollutant.CO2],
        )
        assert len(engine.pollutants) == 2

    def test_passing_rata(self, rata_engine, passing_test_data):
        """Test passing RATA calculation."""
        result = rata_engine.calculate_relative_accuracy(passing_test_data)

        assert result.status == TestStatus.PASSED
        assert result.relative_accuracy_pct <= 10.0
        assert len(result.provenance_hash) == 64

    def test_failing_rata(self, rata_engine, failing_test_data):
        """Test failing RATA calculation."""
        result = rata_engine.calculate_relative_accuracy(failing_test_data)

        assert result.status == TestStatus.FAILED
        assert result.relative_accuracy_pct > 10.0

    def test_ra_formula(self, rata_engine, passing_test_data):
        """Test RA formula calculation."""
        result = rata_engine.calculate_relative_accuracy(passing_test_data)

        # RA = (|d-bar| + |CC|) / RM-bar * 100
        assert result.mean_difference is not None
        assert result.standard_deviation is not None
        assert result.confidence_coefficient is not None
        assert result.mean_reference_value is not None

    def test_bias_analysis(self, rata_engine, passing_test_data):
        """Test bias analysis."""
        result = rata_engine.calculate_relative_accuracy(passing_test_data)

        assert result.bias_status in [
            BiasAdjustmentStatus.NOT_REQUIRED,
            BiasAdjustmentStatus.REQUIRED,
        ]

    def test_diluent_threshold(self, rata_engine):
        """Test O2/CO2 diluent threshold (1% absolute)."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=5.0,  # 5% O2
                reference_unit="%",
                cems_value=4.8,  # 4.8% O2
                load_percent=85.0,
            ))

        test_data = RATATestInput(
            unit_id="UNIT-001",
            pollutant=CEMSPollutant.O2,
            test_date=date.today(),
            reference_method="3A",
            analyzer_span=25.0,
            runs=runs,
        )

        result = rata_engine.calculate_relative_accuracy(test_data)

        # With small absolute difference, should use diluent criteria
        assert result.threshold_used in ["diluent_absolute", "percent"]

    def test_frequency_determination_quarterly(self, rata_engine, passing_test_data):
        """Test quarterly frequency determination."""
        # First RATA
        result = rata_engine.calculate_relative_accuracy(passing_test_data)

        # With only one good RATA, should remain quarterly
        assert result.next_rata_frequency == RATAFrequency.QUARTERLY

    def test_frequency_determination_semiannual(self, rata_engine):
        """Test semiannual frequency after 2 consecutive good RATAs."""
        # Simulate prior good RATAs
        test_data = self._create_good_test_data(rata_engine.unit_id)
        test_data.prior_rata_results = [6.0, 7.0]  # 2 prior good RATAs

        result = rata_engine.calculate_relative_accuracy(test_data)

        # With 2 prior good + 1 current, may qualify for semiannual
        if result.relative_accuracy_pct <= 7.5:
            assert result.next_rata_frequency in [
                RATAFrequency.SEMIANNUAL,
                RATAFrequency.ANNUAL,
            ]

    def test_frequency_determination_annual(self, rata_engine):
        """Test annual frequency after 4 consecutive good RATAs."""
        test_data = self._create_good_test_data(rata_engine.unit_id)
        test_data.prior_rata_results = [5.0, 6.0, 5.5, 6.5]  # 4 prior good RATAs

        result = rata_engine.calculate_relative_accuracy(test_data)

        # With 4 prior good + 1 current, may qualify for annual
        if result.relative_accuracy_pct <= 7.5:
            assert result.next_rata_frequency == RATAFrequency.ANNUAL

    def _create_good_test_data(self, unit_id: str) -> RATATestInput:
        """Helper to create passing test data."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=99.0,  # Very close to reference
                load_percent=85.0,
            ))

        return RATATestInput(
            unit_id=unit_id,
            pollutant=CEMSPollutant.NOX,
            test_date=date.today(),
            reference_method="7E",
            analyzer_span=500.0,
            runs=runs,
        )


class TestQuarterlySchedule:
    """Tests for RATA quarterly scheduling."""

    @pytest.fixture
    def rata_engine(self):
        """Create RATA automation engine."""
        return RATAAutomation(unit_id="UNIT-001")

    def test_schedule_generation(self, rata_engine):
        """Test quarterly schedule generation."""
        schedule = rata_engine.get_quarterly_schedule(year=2025)

        assert schedule.unit_id == "UNIT-001"
        assert schedule.year == 2025
        assert len(schedule.scheduled_tests) > 0

    def test_schedule_quarters(self, rata_engine):
        """Test all quarters are scheduled."""
        schedule = rata_engine.get_quarterly_schedule(year=2025)

        quarters = set(t["quarter"] for t in schedule.scheduled_tests)
        assert quarters == {1, 2, 3, 4}

    def test_schedule_pollutants(self, rata_engine):
        """Test all pollutants are scheduled."""
        schedule = rata_engine.get_quarterly_schedule(year=2025)

        # Should have entries for each pollutant
        pollutants_scheduled = set(t["pollutant"] for t in schedule.scheduled_tests)
        assert len(pollutants_scheduled) == len(rata_engine.pollutants)

    def test_next_due_date(self, rata_engine):
        """Test next due date calculation."""
        schedule = rata_engine.get_quarterly_schedule(year=2025)

        assert schedule.next_due_date is not None
        assert isinstance(schedule.days_until_due, int)


class TestCylinderGasAudit:
    """Tests for Cylinder Gas Audit (CGA)."""

    @pytest.fixture
    def rata_engine(self):
        """Create RATA automation engine."""
        return RATAAutomation(unit_id="UNIT-001")

    def test_passing_cga(self, rata_engine):
        """Test passing CGA."""
        result = rata_engine.perform_cylinder_gas_audit(
            pollutant=CEMSPollutant.NOX,
            cylinder_concentration=400.0,
            cems_response=398.0,  # Within 5% of span
            analyzer_span=500.0,
        )

        assert result.passed is True
        assert result.accuracy_percent < 5.0

    def test_failing_cga(self, rata_engine):
        """Test failing CGA."""
        result = rata_engine.perform_cylinder_gas_audit(
            pollutant=CEMSPollutant.NOX,
            cylinder_concentration=400.0,
            cems_response=350.0,  # >5% of span
            analyzer_span=500.0,
        )

        assert result.passed is False
        assert result.accuracy_percent > 5.0

    def test_cga_provenance(self, rata_engine):
        """Test CGA provenance hash."""
        result = rata_engine.perform_cylinder_gas_audit(
            pollutant=CEMSPollutant.NOX,
            cylinder_concentration=400.0,
            cems_response=398.0,
            analyzer_span=500.0,
        )

        assert len(result.provenance_hash) == 64


class TestCalibrationDrift:
    """Tests for calibration drift checking."""

    @pytest.fixture
    def rata_engine(self):
        """Create RATA automation engine."""
        return RATAAutomation(unit_id="UNIT-001")

    def test_passing_drift(self, rata_engine):
        """Test passing calibration drift."""
        result = rata_engine.check_calibration_drift(
            pollutant=CEMSPollutant.NOX,
            zero_reference=0.0,
            zero_cems_response=5.0,  # Within 2.5% of span
            upscale_reference=400.0,
            upscale_cems_response=395.0,  # Within 2.5% of span
            analyzer_span=500.0,
        )

        assert result.passed is True
        assert result.zero_drift_percent <= 2.5
        assert result.span_drift_percent <= 2.5

    def test_failing_drift(self, rata_engine):
        """Test failing calibration drift."""
        result = rata_engine.check_calibration_drift(
            pollutant=CEMSPollutant.NOX,
            zero_reference=0.0,
            zero_cems_response=20.0,  # >2.5% of span
            upscale_reference=400.0,
            upscale_cems_response=395.0,
            analyzer_span=500.0,
        )

        assert result.passed is False
        assert result.zero_drift_percent > 2.5


class TestComplianceStatus:
    """Tests for compliance status tracking."""

    @pytest.fixture
    def rata_engine(self):
        """Create RATA automation engine."""
        return RATAAutomation(unit_id="UNIT-001")

    def test_status_no_history(self, rata_engine):
        """Test compliance status with no history."""
        status = rata_engine.get_compliance_status()

        assert status["unit_id"] == "UNIT-001"
        for pollutant in rata_engine.pollutants:
            poll_status = status["pollutant_status"][pollutant.value]
            assert poll_status["status"] == "NO_HISTORY"

    def test_status_with_history(self, rata_engine):
        """Test compliance status with RATA history."""
        # Run a RATA first
        test_data = self._create_test_data(rata_engine.unit_id)
        rata_engine.calculate_relative_accuracy(test_data)

        status = rata_engine.get_compliance_status()

        poll_status = status["pollutant_status"]["nox"]
        assert "last_rata_date" in poll_status
        assert "days_since_rata" in poll_status

    def _create_test_data(self, unit_id: str) -> RATATestInput:
        """Helper to create test data."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=98.0,
                load_percent=85.0,
            ))

        return RATATestInput(
            unit_id=unit_id,
            pollutant=CEMSPollutant.NOX,
            test_date=date.today(),
            reference_method="7E",
            analyzer_span=500.0,
            runs=runs,
        )


class TestRATAHistory:
    """Tests for RATA history tracking."""

    @pytest.fixture
    def rata_engine(self):
        """Create RATA automation engine."""
        return RATAAutomation(unit_id="UNIT-001")

    def test_history_tracking(self, rata_engine):
        """Test RATA results are tracked in history."""
        test_data = self._create_test_data(rata_engine.unit_id)

        # Run multiple RATAs
        for _ in range(3):
            rata_engine.calculate_relative_accuracy(test_data)

        history = rata_engine.get_rata_history()

        assert len(history) == 3

    def test_history_filter_by_pollutant(self, rata_engine):
        """Test filtering history by pollutant."""
        test_data = self._create_test_data(rata_engine.unit_id)
        rata_engine.calculate_relative_accuracy(test_data)

        history = rata_engine.get_rata_history(pollutant=CEMSPollutant.NOX)

        assert len(history) == 1
        assert all(r.pollutant == "nox" for r in history)

    def test_history_limit(self, rata_engine):
        """Test history limit parameter."""
        test_data = self._create_test_data(rata_engine.unit_id)

        for _ in range(5):
            rata_engine.calculate_relative_accuracy(test_data)

        history = rata_engine.get_rata_history(limit=3)

        assert len(history) == 3

    def _create_test_data(self, unit_id: str) -> RATATestInput:
        """Helper to create test data."""
        runs = []
        for i in range(9):
            start = datetime.now(timezone.utc) + timedelta(hours=i)
            runs.append(RATARunData(
                run_number=i + 1,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                reference_value=100.0,
                reference_unit="ppm",
                cems_value=98.0,
                load_percent=85.0,
            ))

        return RATATestInput(
            unit_id=unit_id,
            pollutant=CEMSPollutant.NOX,
            test_date=date.today(),
            reference_method="7E",
            analyzer_span=500.0,
            runs=runs,
        )


class TestTValues:
    """Tests for t-distribution values."""

    def test_t_values_coverage(self):
        """Test t-values cover required run counts."""
        t_values = RATAAutomation.T_VALUES

        assert 9 in t_values
        assert 10 in t_values
        assert 11 in t_values
        assert 12 in t_values

    def test_t_values_range(self):
        """Test t-values are in expected range."""
        t_values = RATAAutomation.T_VALUES

        for n, t in t_values.items():
            assert 2.0 <= t <= 2.5  # 95% CI t-values for n-1 df
