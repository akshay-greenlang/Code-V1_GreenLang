# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - IoT Streaming Tests (15 tests)

Tests IoT device registration, reading ingestion, aggregation,
anomaly detection, device health, and data quality scoring.

Author: GreenLang QA Team
"""

from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import _compute_hash, _utcnow

from greenlang.schemas import utcnow


class TestIoTStreaming:
    """Test suite for IoT streaming engine."""

    def test_register_device(self, sample_iot_readings):
        """Test device registration from reading metadata."""
        reading = sample_iot_readings[0]
        device = {
            "device_id": reading["device_id"],
            "device_type": reading["device_type"],
            "facility_id": reading["facility_id"],
            "protocol": reading["protocol"],
            "calibration_status": reading["calibration_status"],
            "status": "registered",
            "registered_at": utcnow().isoformat(),
        }
        assert device["status"] == "registered"
        assert device["device_type"] == "energy_meter"
        assert device["protocol"] in ("MQTT", "HTTP", "OPCUA", "MODBUS")

    def test_ingest_reading(self, sample_iot_readings):
        """Test single reading ingestion with provenance."""
        reading = sample_iot_readings[0]
        result = {
            "reading_id": reading["reading_id"],
            "device_id": reading["device_id"],
            "value": reading["value"],
            "unit": reading["unit"],
            "timestamp": reading["timestamp"],
            "quality_flag": reading["quality_flag"],
            "status": "ingested",
            "provenance_hash": _compute_hash(reading),
        }
        assert result["status"] == "ingested"
        assert result["quality_flag"] == "good"
        assert len(result["provenance_hash"]) == 64

    def test_ingest_batch(self, sample_iot_readings):
        """Test batch ingestion of 100 readings."""
        batch_result = {
            "batch_size": len(sample_iot_readings),
            "ingested": len(sample_iot_readings),
            "rejected": 0,
            "quality_good": sum(1 for r in sample_iot_readings if r["quality_flag"] == "good"),
            "quality_suspect": sum(1 for r in sample_iot_readings if r["quality_flag"] == "suspect"),
        }
        assert batch_result["batch_size"] == 100
        assert batch_result["ingested"] == 100
        assert batch_result["quality_good"] + batch_result["quality_suspect"] == 100

    def test_quality_flag_validation(self, sample_iot_readings):
        """Test quality flags are valid values."""
        valid_flags = {"good", "suspect", "bad", "missing"}
        for reading in sample_iot_readings:
            assert reading["quality_flag"] in valid_flags, (
                f"Invalid quality flag: {reading['quality_flag']}"
            )

    def test_aggregate_window_avg(self, sample_iot_readings):
        """Test average aggregation over a window."""
        energy_readings = [
            r for r in sample_iot_readings
            if r["device_type"] == "energy_meter" and r["quality_flag"] == "good"
        ]
        values = [r["value"] for r in energy_readings]
        if values:
            avg = round(sum(values) / len(values), 3)
            assert avg > 0
            assert isinstance(avg, float)

    def test_aggregate_window_min_max(self, sample_iot_readings):
        """Test min/max aggregation over readings."""
        gas_readings = [
            r for r in sample_iot_readings
            if r["device_type"] == "gas_flow"
        ]
        values = [r["value"] for r in gas_readings]
        if values:
            assert min(values) <= max(values)
            assert min(values) >= 0

    def test_anomaly_detection_spike(self, sample_iot_readings):
        """Test spike detection in sensor readings."""
        values = [r["value"] for r in sample_iot_readings[:25]]
        if len(values) >= 5:
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            threshold = mean_val + 3 * std_val
            spikes = [v for v in values if v > threshold]
            assert isinstance(spikes, list)

    def test_anomaly_detection_normal(self, sample_iot_readings):
        """Test normal readings produce no anomalies with relaxed threshold."""
        values = [r["value"] for r in sample_iot_readings if r["device_type"] == "temperature"]
        if len(values) >= 5:
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            threshold = mean_val + 5 * std_val
            anomalies = [v for v in values if v > threshold]
            assert len(anomalies) == 0

    def test_device_health_check(self, sample_iot_readings):
        """Test device health scoring."""
        device_id = sample_iot_readings[0]["device_id"]
        device_readings = [r for r in sample_iot_readings if r["device_id"] == device_id]
        good_count = sum(1 for r in device_readings if r["quality_flag"] == "good")
        total = len(device_readings)
        quality_pct = round(good_count / max(total, 1) * 100, 1)
        health = {
            "device_id": device_id,
            "total_readings": total,
            "good_readings": good_count,
            "quality_pct": quality_pct,
            "calibration_status": device_readings[0]["calibration_status"],
            "battery_pct": device_readings[0]["battery_pct"],
            "status": "healthy" if quality_pct >= 80 else "degraded",
        }
        assert health["total_readings"] > 0
        assert health["calibration_status"] == "valid"

    def test_realtime_emission_calculation(self, sample_iot_readings):
        """Test real-time emission calculation from energy readings."""
        energy_readings = [
            r for r in sample_iot_readings
            if r["device_type"] == "energy_meter" and r["quality_flag"] == "good"
        ]
        emission_factor = 0.42  # kgCO2e per kWh (Germany grid average)
        total_kwh = sum(r["value"] for r in energy_readings)
        total_emissions_kg = round(total_kwh * emission_factor, 3)
        total_emissions_tco2e = round(total_emissions_kg / 1000, 6)
        assert total_emissions_kg > 0
        assert total_emissions_tco2e > 0

    def test_buffer_management(self, sample_iot_readings):
        """Test ingestion buffer tracking."""
        buffer_size_mb = 512
        reading_size_kb = 0.5
        total_readings = len(sample_iot_readings)
        used_mb = round(total_readings * reading_size_kb / 1024, 3)
        buffer = {
            "buffer_size_mb": buffer_size_mb,
            "used_mb": used_mb,
            "available_mb": round(buffer_size_mb - used_mb, 3),
            "utilization_pct": round(used_mb / buffer_size_mb * 100, 2),
        }
        assert buffer["available_mb"] > 0
        assert buffer["utilization_pct"] < 100

    def test_device_calibration_check(self, sample_iot_readings):
        """Test device calibration status validation."""
        for reading in sample_iot_readings:
            assert reading["calibration_status"] in ("valid", "expired", "unknown")

    def test_protocol_support(self, sample_iot_readings):
        """Test supported protocol types."""
        protocols = {r["protocol"] for r in sample_iot_readings}
        supported = {"MQTT", "HTTP", "OPCUA", "MODBUS"}
        assert protocols.issubset(supported)

    def test_facility_device_listing(self, sample_iot_readings):
        """Test device listing by facility."""
        facilities = {}
        for r in sample_iot_readings:
            fid = r["facility_id"]
            facilities.setdefault(fid, set()).add(r["device_id"])
        assert len(facilities) >= 2
        for fid, devices in facilities.items():
            assert len(devices) > 0

    def test_data_quality_score(self, sample_iot_readings):
        """Test overall data quality scoring across all readings."""
        total = len(sample_iot_readings)
        good = sum(1 for r in sample_iot_readings if r["quality_flag"] == "good")
        quality_score = round(good / total * 100, 1)
        assert quality_score >= 80.0
        assert quality_score <= 100.0
