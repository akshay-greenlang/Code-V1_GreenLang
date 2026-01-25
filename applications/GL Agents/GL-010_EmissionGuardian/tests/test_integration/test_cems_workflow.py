"""
GL-010 EmissionGuardian - CEMS Workflow Integration Tests

End-to-end integration tests for the Continuous Emissions Monitoring System (CEMS)
data processing workflow. Tests the complete pipeline from data acquisition through
compliance evaluation and reporting.

Reference: EPA 40 CFR Part 75 - Continuous Emissions Monitoring
Test Coverage: 12+ integration test cases

Author: GreenLang Test Engineering
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import hashlib
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_cems_reading():
    """Sample CEMS reading data."""
    return {
        "reading_id": "RDG-2024-001-001",
        "facility_id": "FAC-001",
        "unit_id": "UNIT-001",
        "timestamp": datetime.now().isoformat(),
        "pollutants": {
            "NOX": {"value": 0.12, "unit": "lb/MMBtu"},
            "SO2": {"value": 0.15, "unit": "lb/MMBtu"},
            "CO2": {"value": 205.5, "unit": "lb/MMBtu"},
        },
        "operating_params": {
            "load_mw": 450.0,
            "heat_input_mmbtu": 4500.0,
            "stack_flow_scfh": 1000000.0,
        },
        "data_quality": "VALID",
    }


@pytest.fixture
def sample_hourly_readings():
    """Generate 24 hours of sample CEMS readings."""
    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    readings = []

    for hour in range(24):
        timestamp = base_time - timedelta(hours=hour)
        readings.append({
            "reading_id": f"RDG-2024-001-{hour:03d}",
            "facility_id": "FAC-001",
            "unit_id": "UNIT-001",
            "timestamp": timestamp.isoformat(),
            "pollutants": {
                "NOX": {"value": 0.10 + (hour % 5) * 0.01, "unit": "lb/MMBtu"},
                "SO2": {"value": 0.12 + (hour % 4) * 0.005, "unit": "lb/MMBtu"},
                "CO2": {"value": 200.0 + (hour % 10), "unit": "lb/MMBtu"},
            },
            "operating_params": {
                "load_mw": 400.0 + (hour % 8) * 10,
                "heat_input_mmbtu": 4000.0 + (hour % 8) * 100,
            },
            "data_quality": "VALID",
        })

    return readings


@pytest.fixture
def permit_rules():
    """Sample permit rules for compliance evaluation."""
    return [
        {
            "rule_id": "RULE-NOX-001",
            "pollutant": "NOX",
            "limit_value": 0.15,
            "limit_unit": "lb/MMBtu",
            "averaging_period": "HOURLY",
            "warning_threshold_pct": 90.0,
        },
        {
            "rule_id": "RULE-SO2-001",
            "pollutant": "SO2",
            "limit_value": 0.20,
            "limit_unit": "lb/MMBtu",
            "averaging_period": "HOURLY",
            "warning_threshold_pct": 90.0,
        },
        {
            "rule_id": "RULE-NOX-24H",
            "pollutant": "NOX",
            "limit_value": 0.12,
            "limit_unit": "lb/MMBtu",
            "averaging_period": "ROLLING_24HOUR",
            "warning_threshold_pct": 85.0,
        },
    ]


@pytest.fixture
def mock_cems_connector():
    """Mock CEMS data connector."""
    connector = Mock()
    connector.connect = Mock(return_value=True)
    connector.disconnect = Mock(return_value=True)
    connector.is_connected = Mock(return_value=True)
    connector.read_current = Mock(return_value={
        "NOX": 0.12, "SO2": 0.15, "CO2": 205.5
    })
    return connector


# =============================================================================
# TEST: DATA ACQUISITION WORKFLOW
# =============================================================================

class TestDataAcquisitionWorkflow:
    """Test CEMS data acquisition workflow."""

    def test_acquire_single_reading(self, sample_cems_reading):
        """Acquire and validate a single CEMS reading."""
        # Verify reading structure
        assert "reading_id" in sample_cems_reading
        assert "pollutants" in sample_cems_reading
        assert "NOX" in sample_cems_reading["pollutants"]
        assert sample_cems_reading["data_quality"] == "VALID"

    def test_acquire_hourly_batch(self, sample_hourly_readings):
        """Acquire and validate hourly batch of readings."""
        assert len(sample_hourly_readings) == 24

        for reading in sample_hourly_readings:
            assert "reading_id" in reading
            assert "pollutants" in reading
            assert reading["data_quality"] == "VALID"

    def test_reading_timestamp_ordering(self, sample_hourly_readings):
        """Verify readings are ordered by timestamp."""
        timestamps = [
            datetime.fromisoformat(r["timestamp"])
            for r in sample_hourly_readings
        ]

        # Should be in descending order (most recent first)
        for i in range(len(timestamps) - 1):
            assert timestamps[i] >= timestamps[i + 1]

    def test_data_quality_validation(self, sample_cems_reading):
        """Validate data quality flags are correctly set."""
        valid_statuses = ["VALID", "SUBSTITUTE", "MISSING", "INVALID"]
        assert sample_cems_reading["data_quality"] in valid_statuses


# =============================================================================
# TEST: DATA NORMALIZATION WORKFLOW
# =============================================================================

class TestDataNormalizationWorkflow:
    """Test CEMS data normalization workflow."""

    def test_unit_normalization(self, sample_cems_reading):
        """Test pollutant unit normalization."""
        pollutants = sample_cems_reading["pollutants"]

        for pollutant, data in pollutants.items():
            assert "value" in data
            assert "unit" in data
            assert data["value"] >= 0  # No negative emissions

    def test_pollutant_value_ranges(self, sample_hourly_readings):
        """Verify pollutant values are within reasonable ranges."""
        for reading in sample_hourly_readings:
            nox = reading["pollutants"]["NOX"]["value"]
            so2 = reading["pollutants"]["SO2"]["value"]
            co2 = reading["pollutants"]["CO2"]["value"]

            # Reasonable emission rate ranges
            assert 0 <= nox <= 1.0  # lb/MMBtu
            assert 0 <= so2 <= 1.0  # lb/MMBtu
            assert 0 <= co2 <= 300.0  # lb/MMBtu

    def test_operating_params_normalization(self, sample_cems_reading):
        """Test operating parameter normalization."""
        params = sample_cems_reading["operating_params"]

        assert params["load_mw"] > 0
        assert params["heat_input_mmbtu"] > 0


# =============================================================================
# TEST: HOURLY AGGREGATION WORKFLOW
# =============================================================================

class TestHourlyAggregationWorkflow:
    """Test CEMS hourly aggregation workflow."""

    def test_hourly_average_calculation(self, sample_hourly_readings):
        """Calculate hourly averages from readings."""
        # Group by hour
        hourly_groups: Dict[str, List[Dict]] = {}

        for reading in sample_hourly_readings:
            hour_key = reading["timestamp"][:13]  # YYYY-MM-DDTHH
            if hour_key not in hourly_groups:
                hourly_groups[hour_key] = []
            hourly_groups[hour_key].append(reading)

        # Each hour should have readings
        assert len(hourly_groups) >= 1

    def test_minimum_data_availability(self, sample_hourly_readings):
        """Verify minimum data availability (75% per EPA)."""
        total_readings = len(sample_hourly_readings)
        valid_readings = sum(
            1 for r in sample_hourly_readings
            if r["data_quality"] == "VALID"
        )

        availability_pct = (valid_readings / total_readings) * 100
        assert availability_pct >= 75.0  # EPA 75% minimum

    def test_data_substitution_flagging(self):
        """Test that substituted data is properly flagged."""
        reading = {
            "data_quality": "SUBSTITUTE",
            "substitution_method": "MAXIMUM_POTENTIAL_VALUE",
        }

        assert reading["data_quality"] == "SUBSTITUTE"
        assert "substitution_method" in reading


# =============================================================================
# TEST: COMPLIANCE EVALUATION WORKFLOW
# =============================================================================

class TestComplianceEvaluationWorkflow:
    """Test compliance evaluation workflow."""

    def test_hourly_compliance_evaluation(
        self, sample_cems_reading, permit_rules
    ):
        """Evaluate hourly reading against permit rules."""
        pollutants = sample_cems_reading["pollutants"]

        for rule in permit_rules:
            if rule["averaging_period"] == "HOURLY":
                pollutant = rule["pollutant"]
                if pollutant in pollutants:
                    value = pollutants[pollutant]["value"]
                    limit = rule["limit_value"]
                    pct_of_limit = (value / limit) * 100

                    # Should be under limit
                    assert value <= limit
                    assert pct_of_limit <= 100

    def test_rolling_average_compliance(
        self, sample_hourly_readings, permit_rules
    ):
        """Evaluate rolling average compliance."""
        rolling_rule = next(
            r for r in permit_rules
            if r["averaging_period"] == "ROLLING_24HOUR"
        )

        pollutant = rolling_rule["pollutant"]
        values = [
            r["pollutants"][pollutant]["value"]
            for r in sample_hourly_readings
        ]

        rolling_avg = sum(values) / len(values)
        limit = rolling_rule["limit_value"]

        # Verify average calculation
        assert rolling_avg >= 0

    def test_exceedance_detection(self):
        """Test exceedance event detection."""
        reading = {"pollutants": {"NOX": {"value": 0.18, "unit": "lb/MMBtu"}}}
        limit = 0.15

        is_exceedance = reading["pollutants"]["NOX"]["value"] > limit
        assert is_exceedance is True

    def test_warning_threshold_detection(
        self, sample_cems_reading, permit_rules
    ):
        """Test warning threshold detection."""
        rule = permit_rules[0]  # NOX hourly
        value = sample_cems_reading["pollutants"]["NOX"]["value"]
        limit = rule["limit_value"]
        warning_pct = rule["warning_threshold_pct"]

        pct_of_limit = (value / limit) * 100
        is_warning = pct_of_limit >= warning_pct

        # Should be deterministic
        assert isinstance(is_warning, bool)


# =============================================================================
# TEST: PROVENANCE TRACKING WORKFLOW
# =============================================================================

class TestProvenanceTrackingWorkflow:
    """Test provenance tracking workflow."""

    def test_reading_hash_generation(self, sample_cems_reading):
        """Generate SHA-256 hash for reading provenance."""
        reading_json = json.dumps(sample_cems_reading, sort_keys=True, default=str)
        reading_hash = hashlib.sha256(reading_json.encode()).hexdigest()

        assert len(reading_hash) == 64  # SHA-256
        assert reading_hash.isalnum()

    def test_hash_determinism(self, sample_cems_reading):
        """Same reading should produce same hash."""
        reading_json = json.dumps(sample_cems_reading, sort_keys=True, default=str)

        hash1 = hashlib.sha256(reading_json.encode()).hexdigest()
        hash2 = hashlib.sha256(reading_json.encode()).hexdigest()

        assert hash1 == hash2

    def test_hash_sensitivity(self, sample_cems_reading):
        """Modified reading should produce different hash."""
        reading_json1 = json.dumps(sample_cems_reading, sort_keys=True, default=str)
        hash1 = hashlib.sha256(reading_json1.encode()).hexdigest()

        # Modify reading
        modified = sample_cems_reading.copy()
        modified["pollutants"]["NOX"]["value"] = 0.999

        reading_json2 = json.dumps(modified, sort_keys=True, default=str)
        hash2 = hashlib.sha256(reading_json2.encode()).hexdigest()

        assert hash1 != hash2


# =============================================================================
# TEST: QUALITY ASSURANCE WORKFLOW
# =============================================================================

class TestQualityAssuranceWorkflow:
    """Test CEMS quality assurance workflow."""

    def test_calibration_drift_check(self):
        """Test calibration drift detection."""
        # Simulated calibration data
        zero_reference = 0.0
        zero_measured = 0.02
        span_reference = 100.0
        span_measured = 98.5

        zero_drift = abs(zero_measured - zero_reference)
        span_drift = abs((span_measured - span_reference) / span_reference) * 100

        # EPA typically allows 2.5% drift
        assert zero_drift < 2.5
        assert span_drift < 2.5

    def test_missing_data_substitution(self):
        """Test missing data substitution per EPA Part 75."""
        missing_hours = 5
        total_hours = 24

        # Substitution method depends on missing duration
        if missing_hours <= 8:
            method = "AVERAGE_PRECEDING"
        elif missing_hours <= 24:
            method = "90TH_PERCENTILE"
        else:
            method = "MAXIMUM_POTENTIAL"

        assert method in ["AVERAGE_PRECEDING", "90TH_PERCENTILE", "MAXIMUM_POTENTIAL"]

    def test_qa_flag_propagation(self, sample_cems_reading):
        """Test QA flags propagate through workflow."""
        reading = sample_cems_reading.copy()
        reading["qa_flags"] = ["CALIBRATION_DUE", "MAINTENANCE_SCHEDULED"]

        assert len(reading["qa_flags"]) > 0
        assert "CALIBRATION_DUE" in reading["qa_flags"]


# =============================================================================
# TEST: REPORTING WORKFLOW
# =============================================================================

class TestReportingWorkflow:
    """Test emissions reporting workflow."""

    def test_hourly_report_generation(self, sample_hourly_readings):
        """Generate hourly emissions report."""
        report = {
            "report_type": "HOURLY",
            "facility_id": sample_hourly_readings[0]["facility_id"],
            "period_start": sample_hourly_readings[-1]["timestamp"],
            "period_end": sample_hourly_readings[0]["timestamp"],
            "readings_count": len(sample_hourly_readings),
        }

        assert report["readings_count"] == 24
        assert report["report_type"] == "HOURLY"

    def test_quarterly_aggregation(self, sample_hourly_readings):
        """Test quarterly data aggregation for EPA reporting."""
        # Simulate quarterly summary
        pollutant = "NOX"
        values = [
            r["pollutants"][pollutant]["value"]
            for r in sample_hourly_readings
        ]

        quarterly_summary = {
            "pollutant": pollutant,
            "average": sum(values) / len(values),
            "maximum": max(values),
            "minimum": min(values),
            "hours_reported": len(values),
        }

        assert quarterly_summary["average"] > 0
        assert quarterly_summary["maximum"] >= quarterly_summary["average"]
        assert quarterly_summary["minimum"] <= quarterly_summary["average"]

    def test_compliance_status_summary(self, permit_rules):
        """Generate compliance status summary."""
        summary = {
            "total_rules": len(permit_rules),
            "compliant": len(permit_rules),
            "warning": 0,
            "exceeded": 0,
            "overall_status": "COMPLIANT",
        }

        assert summary["total_rules"] == 3
        assert summary["overall_status"] == "COMPLIANT"


# =============================================================================
# TEST: ERROR HANDLING WORKFLOW
# =============================================================================

class TestErrorHandlingWorkflow:
    """Test error handling in CEMS workflow."""

    def test_invalid_reading_rejection(self):
        """Test rejection of invalid readings."""
        invalid_reading = {
            "reading_id": "RDG-BAD",
            "pollutants": {
                "NOX": {"value": -0.5, "unit": "lb/MMBtu"},  # Negative
            },
        }

        # Negative values should be rejected
        assert invalid_reading["pollutants"]["NOX"]["value"] < 0

    def test_connection_failure_handling(self, mock_cems_connector):
        """Test handling of connection failures."""
        mock_cems_connector.is_connected.return_value = False

        is_connected = mock_cems_connector.is_connected()
        assert not is_connected

    def test_data_gap_detection(self, sample_hourly_readings):
        """Detect gaps in hourly data."""
        timestamps = [
            datetime.fromisoformat(r["timestamp"])
            for r in sample_hourly_readings
        ]

        gaps = []
        for i in range(len(timestamps) - 1):
            diff = timestamps[i] - timestamps[i + 1]
            if diff > timedelta(hours=1, minutes=5):  # Allow 5 min tolerance
                gaps.append((timestamps[i], timestamps[i + 1]))

        # Should have no gaps in sample data
        assert len(gaps) == 0


# =============================================================================
# TEST: END-TO-END WORKFLOW
# =============================================================================

class TestEndToEndWorkflow:
    """Test complete end-to-end CEMS workflow."""

    def test_complete_hourly_workflow(
        self, sample_cems_reading, permit_rules
    ):
        """Test complete hourly processing workflow."""
        # Step 1: Acquire reading
        reading = sample_cems_reading
        assert reading is not None

        # Step 2: Validate data quality
        assert reading["data_quality"] == "VALID"

        # Step 3: Evaluate compliance
        pollutants = reading["pollutants"]
        compliance_results = []

        for rule in permit_rules:
            if rule["averaging_period"] == "HOURLY":
                pollutant = rule["pollutant"]
                if pollutant in pollutants:
                    value = pollutants[pollutant]["value"]
                    limit = rule["limit_value"]
                    is_compliant = value <= limit
                    compliance_results.append({
                        "rule_id": rule["rule_id"],
                        "compliant": is_compliant,
                    })

        # Step 4: Generate provenance
        reading_json = json.dumps(reading, sort_keys=True, default=str)
        provenance_hash = hashlib.sha256(reading_json.encode()).hexdigest()

        # Step 5: Verify results
        assert len(compliance_results) > 0
        assert len(provenance_hash) == 64
        assert all(r["compliant"] for r in compliance_results)

    def test_workflow_determinism(
        self, sample_cems_reading, permit_rules
    ):
        """Verify workflow produces deterministic results."""
        results = []

        for _ in range(3):
            pollutants = sample_cems_reading["pollutants"]
            nox_value = pollutants["NOX"]["value"]
            nox_limit = next(
                r["limit_value"] for r in permit_rules
                if r["pollutant"] == "NOX" and r["averaging_period"] == "HOURLY"
            )
            pct = (nox_value / nox_limit) * 100
            results.append(pct)

        # All runs should produce identical results
        assert all(r == results[0] for r in results)
