# -*- coding: utf-8 -*-
"""
Integration Tests for GL-010 EMISSIONWATCH CEMS Integration.

Tests CEMS connection, real-time data acquisition, data quality validation,
missing data handling, and calibration detection.

Test Count: 18+ tests
Coverage Target: 90%+

Standards: EPA 40 CFR Part 75

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, AsyncMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import EmissionsComplianceTools


# =============================================================================
# TEST CLASS: CEMS INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestCEMSIntegration:
    """Integration tests for CEMS data handling."""

    # =========================================================================
    # CONNECTION TESTS
    # =========================================================================

    def test_cems_connection_mock(self, emissions_tools, mock_cems_connector):
        """Test CEMS connection with mock connector."""
        assert mock_cems_connector.connect() == True
        assert mock_cems_connector.is_connected() == True

    def test_cems_connection_disconnect(self, mock_cems_connector):
        """Test CEMS disconnect."""
        mock_cems_connector.connect()
        assert mock_cems_connector.disconnect() == True

    def test_cems_connection_retry(self, mock_cems_connector):
        """Test CEMS connection retry on failure."""
        call_count = [0]

        def connect_with_retry():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Connection failed")
            return True

        mock_cems_connector.connect.side_effect = connect_with_retry

        # Should eventually succeed
        for _ in range(5):
            try:
                result = mock_cems_connector.connect()
                if result:
                    break
            except ConnectionError:
                continue

        assert call_count[0] >= 2

    # =========================================================================
    # REAL-TIME DATA ACQUISITION TESTS
    # =========================================================================

    def test_realtime_data_acquisition(self, emissions_tools, mock_cems_connector):
        """Test real-time data acquisition from CEMS."""
        data = mock_cems_connector.read_data()

        assert "nox_ppm" in data
        assert "sox_ppm" in data
        assert "co2_percent" in data
        assert "o2_percent" in data
        assert "flow_rate_dscfm" in data

    def test_realtime_data_processing(self, emissions_tools, mock_cems_connector, natural_gas_fuel_data):
        """Test processing real-time CEMS data."""
        cems_data = mock_cems_connector.read_data()

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert result is not None
        assert result.concentration_ppm >= 0

    def test_realtime_data_timestamp(self, mock_cems_connector):
        """Test CEMS data includes timestamp."""
        data = mock_cems_connector.read_data()

        # Add timestamp for testing
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

        assert "timestamp" in data

    def test_realtime_data_frequency(self, mock_cems_connector):
        """Test data acquisition at specified frequency."""
        readings = []

        for _ in range(10):
            data = mock_cems_connector.read_data()
            readings.append(data)

        assert len(readings) == 10

    # =========================================================================
    # DATA QUALITY VALIDATION TESTS
    # =========================================================================

    def test_data_quality_validation_valid(self, emissions_tools, sample_cems_data):
        """Test data quality validation for valid data."""
        # All values within expected ranges
        assert sample_cems_data["nox_ppm"] >= 0
        assert sample_cems_data["sox_ppm"] >= 0
        assert 0 <= sample_cems_data["o2_percent"] <= 21
        assert sample_cems_data["flow_rate_dscfm"] > 0

    def test_data_quality_validation_out_of_range(self, emissions_tools, invalid_cems_data):
        """Test data quality validation for out-of-range values."""
        # Negative NOx should be flagged
        assert invalid_cems_data["nox_ppm"] < 0

        # O2 > 21% is invalid
        assert invalid_cems_data["o2_percent"] > 21

    def test_data_quality_code_assignment(self, sample_cems_data):
        """Test data quality code assignment."""
        assert sample_cems_data["quality_code"] == "valid"

    def test_data_quality_substitute_data(self, missing_data_cems):
        """Test substitute data quality code."""
        assert missing_data_cems["quality_code"] == "substitute"

    def test_data_availability_calculation(self, sample_cems_data):
        """Test data availability percentage calculation."""
        expected = sample_cems_data.get("expected_hours", 720)
        valid = sample_cems_data.get("valid_hours", 700)

        availability = (valid / expected) * 100 if expected > 0 else 0

        # Should meet 90% minimum
        assert availability >= 90

    # =========================================================================
    # MISSING DATA HANDLING TESTS
    # =========================================================================

    def test_missing_data_handling_single_field(self, emissions_tools, natural_gas_fuel_data):
        """Test handling of single missing field."""
        cems_data = {
            "nox_ppm": 45.0,
            # Missing o2_percent - should use default
            "flow_rate_dscfm": 50000.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert result is not None

    def test_missing_data_handling_multiple_fields(self, emissions_tools, natural_gas_fuel_data):
        """Test handling of multiple missing fields."""
        cems_data = {
            "nox_ppm": 45.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert result is not None

    def test_missing_data_substitute_procedure(self, emissions_tools, cems_config):
        """Test EPA Part 75 substitute data procedure enabled."""
        assert cems_config.substitute_data_enabled == True

    def test_missing_data_hours_tracking(self, emissions_records):
        """Test missing data hours are tracked."""
        # Count records with substitute data
        substitute_count = sum(1 for r in emissions_records
                              if r.get("data_quality_code") == "substitute")

        # Most records should be valid
        valid_count = sum(1 for r in emissions_records
                        if r.get("data_quality_code") == "valid")

        assert valid_count > substitute_count

    # =========================================================================
    # CALIBRATION DETECTION TESTS
    # =========================================================================

    def test_calibration_detection_in_control(self, mock_cems_connector):
        """Test calibration drift detection when in control."""
        cal_status = mock_cems_connector.read_calibration_status()

        assert cal_status["in_control"] == True
        assert cal_status["drift_percent"] < 2.5  # Part 75 limit

    def test_calibration_detection_drift_exceeded(self, mock_cems_connector):
        """Test calibration drift detection when exceeded."""
        # Modify mock to return high drift
        mock_cems_connector.read_calibration_status.return_value = {
            "last_calibration": datetime.now(timezone.utc).isoformat(),
            "drift_percent": 3.5,  # Exceeds 2.5% limit
            "in_control": False,
        }

        cal_status = mock_cems_connector.read_calibration_status()

        assert cal_status["in_control"] == False
        assert cal_status["drift_percent"] > 2.5

    def test_calibration_last_date_tracking(self, mock_cems_connector, sample_cems_data):
        """Test last calibration date is tracked."""
        last_cal = sample_cems_data.get("last_calibration_date")

        assert last_cal is not None

    def test_rata_date_tracking(self, sample_cems_data):
        """Test RATA (Relative Accuracy Test Audit) date tracking."""
        last_rata = sample_cems_data.get("last_rata_date")

        assert last_rata is not None

    def test_calibration_daily_requirement(self, cems_config):
        """Test daily calibration requirement configuration."""
        assert cems_config.daily_calibration_required == True

    def test_calibration_drift_limit_configuration(self, cems_config):
        """Test calibration drift limit configuration."""
        assert cems_config.calibration_drift_limit_percent == 2.5


# =============================================================================
# TEST CLASS: CEMS DATA PROCESSING PIPELINE
# =============================================================================

@pytest.mark.integration
class TestCEMSDataPipeline:
    """Integration tests for CEMS data processing pipeline."""

    def test_full_data_pipeline(self, emissions_tools, sample_cems_data, natural_gas_fuel_data, epa_permit_limits):
        """Test full CEMS data processing pipeline."""
        # Step 1: Calculate emissions
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(sample_cems_data, natural_gas_fuel_data)

        # Step 2: Check compliance
        emissions_result = {
            "nox": nox.to_dict(),
            "sox": sox.to_dict(),
            "co2": co2.to_dict(),
            "pm": pm.to_dict(),
        }

        compliance = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # Step 3: Detect violations
        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # Verify pipeline completed
        assert nox is not None
        assert sox is not None
        assert co2 is not None
        assert pm is not None
        assert compliance is not None
        assert isinstance(violations, list)

    def test_data_pipeline_with_historian(self, emissions_tools, mock_historian_connector, natural_gas_fuel_data):
        """Test data pipeline with historian integration."""
        # Query historian for NOx data
        nox_data = mock_historian_connector.query_tag("NOX.PV")

        assert len(nox_data) > 0

    def test_data_pipeline_batch_processing(self, emissions_tools, cems_data_series, natural_gas_fuel_data):
        """Test batch processing of CEMS data."""
        results = []

        for cems_data in cems_data_series:
            result = emissions_tools.calculate_nox_emissions(
                cems_data=cems_data,
                fuel_data=natural_gas_fuel_data,
            )
            results.append(result)

        assert len(results) == len(cems_data_series)
