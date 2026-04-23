"""GL-013 Data Quality Tests - Author: GL-TestEngineer"""
import pytest
from datetime import datetime
import math

class TestTimestampValidation:
    def test_valid_timestamp_format(self, sample_good_quality_data):
        timestamp = sample_good_quality_data["timestamp"]
        assert timestamp is not None
        # Should be parseable
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_invalid_timestamp_detection(self, sample_bad_quality_data):
        timestamp = sample_bad_quality_data["timestamp"]
        with pytest.raises(Exception):
            datetime.fromisoformat(timestamp)

    def test_timestamp_not_future(self, sample_good_quality_data):
        timestamp = sample_good_quality_data["timestamp"]
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert dt <= datetime.now(dt.tzinfo)

class TestUnitConsistencyChecks:
    def test_valid_vibration_unit(self, sample_good_quality_data):
        unit = sample_good_quality_data["unit"]
        valid_units = ["g", "mm/s", "m/s2", "ips", "mils"]
        assert unit in valid_units

    def test_invalid_unit_detection(self, sample_bad_quality_data):
        unit = sample_bad_quality_data["unit"]
        valid_units = ["g", "mm/s", "m/s2", "ips", "mils"]
        assert unit not in valid_units

    def test_sample_rate_positive(self, sample_good_quality_data):
        sample_rate = sample_good_quality_data["sample_rate_hz"]
        assert sample_rate > 0

    def test_invalid_sample_rate(self, sample_bad_quality_data):
        sample_rate = sample_bad_quality_data["sample_rate_hz"]
        assert sample_rate <= 0

class TestSensorHealthDetection:
    def test_good_data_passes_health_check(self, sample_good_quality_data):
        flags = sample_good_quality_data["quality_flags"]
        assert flags["sensor_healthy"] == True
        assert flags["timestamp_valid"] == True

    def test_bad_data_fails_health_check(self, sample_bad_quality_data):
        flags = sample_bad_quality_data["quality_flags"]
        assert flags["sensor_healthy"] == False

class TestMissingValueDetection:
    def test_no_missing_values_good_data(self, sample_good_quality_data):
        values = sample_good_quality_data["values"]
        assert None not in values
        assert all(not math.isnan(v) for v in values)

    def test_missing_values_detected(self, sample_bad_quality_data):
        values = sample_bad_quality_data["values"]
        has_missing = None in values or any(
            isinstance(v, float) and math.isnan(v) for v in values if v is not None
        )
        assert has_missing

class TestValueRangeValidation:
    def test_vibration_values_in_range(self, sample_vibration_readings):
        for reading in sample_vibration_readings:
            assert -100 < reading.value < 100  # Reasonable g range

    def test_temperature_values_in_range(self, sample_temperature_readings):
        for reading in sample_temperature_readings:
            assert -40 < reading.value < 200  # Reasonable temp range

    def test_pressure_values_positive(self, sample_pressure_readings):
        for reading in sample_pressure_readings:
            assert reading.value > 0

class TestDataGapDetection:
    def test_no_data_gaps_good_data(self, sample_vibration_readings):
        timestamps = [r.timestamp for r in sample_vibration_readings]
        if len(timestamps) > 1:
            gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps)-1)]
            max_gap = max(gaps) if gaps else 0
            assert max_gap < 1.0  # No gaps > 1 second

class TestQualityScoreCalculation:
    def test_quality_score_bounded(self, sample_vibration_readings):
        for reading in sample_vibration_readings:
            assert 0 <= reading.quality_score <= 1

    def test_high_quality_readings(self, sample_vibration_readings):
        avg_quality = sum(r.quality_score for r in sample_vibration_readings) / len(sample_vibration_readings)
        assert avg_quality > 0.9

class TestSensorMetadataValidation:
    def test_sensor_id_present(self, sample_vibration_readings):
        for reading in sample_vibration_readings:
            assert reading.sensor_id is not None
            assert len(reading.sensor_id) > 0

    def test_sensor_type_valid(self, sample_vibration_readings):
        valid_types = ["vibration_acceleration", "vibration_velocity", "temperature", "current", "pressure"]
        for reading in sample_vibration_readings:
            assert reading.sensor_type in valid_types
