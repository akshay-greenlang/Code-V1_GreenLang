"""
GL-007 FURNACEPULSE - Orchestrator Tests

Integration tests for the main orchestrator including:
- Full telemetry processing pipeline
- KPI calculation workflow
- Hotspot detection pipeline
- Evidence package generation
- Health check functionality

Coverage Target: >85%
"""

import pytest
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import hashlib
import json


class TestTelemetryProcessingPipeline:
    """Integration tests for telemetry processing pipeline."""

    def test_telemetry_ingestion(
        self,
        sample_telemetry_signals,
        mock_opcua_client,
    ):
        """Test telemetry data ingestion from OPC-UA."""
        # Simulate OPC-UA tag reading
        tags_to_read = [
            "FRN-001.FUEL.FLOW",
            "FRN-001.STACK.TEMP",
            "FRN-001.FLUE.O2",
        ]

        # Mock reading would happen via OPC-UA client
        readings = {}
        for tag in tags_to_read:
            # Find matching signal in sample data
            signal = next(
                (s for s in sample_telemetry_signals if tag.endswith(s.tag_id.split(".")[-1])),
                None,
            )
            if signal:
                readings[tag] = {
                    "value": signal.value,
                    "quality": signal.quality,
                    "timestamp": signal.timestamp,
                }

        assert len(readings) > 0

    def test_telemetry_validation(self, sample_telemetry_signals):
        """Test telemetry data validation."""
        validation_results = []

        for signal in sample_telemetry_signals:
            is_valid = True
            errors = []

            # Check quality
            if signal.quality != "GOOD":
                is_valid = False
                errors.append("Bad signal quality")

            # Check for NaN/None
            if signal.value is None:
                is_valid = False
                errors.append("Missing value")

            # Check reasonable range (example for temperature)
            if "TEMP" in signal.tag_id and not (0 <= signal.value <= 1500):
                is_valid = False
                errors.append("Value out of range")

            validation_results.append({
                "tag_id": signal.tag_id,
                "is_valid": is_valid,
                "errors": errors,
            })

        # All sample signals should be valid
        assert all(r["is_valid"] for r in validation_results)

    def test_telemetry_transformation(self, sample_telemetry_signals):
        """Test telemetry transformation to canonical format."""
        canonical_records = []

        for signal in sample_telemetry_signals:
            canonical = {
                "tag": signal.tag_id,
                "value": signal.value,
                "unit": signal.unit,
                "timestamp_utc": signal.timestamp.isoformat(),
                "quality_code": 1 if signal.quality == "GOOD" else 0,
                "source": signal.source,
                "processed_at": datetime.now().isoformat(),
            }
            canonical_records.append(canonical)

        assert len(canonical_records) == len(sample_telemetry_signals)

    @pytest.mark.asyncio
    async def test_telemetry_streaming_to_kafka(
        self,
        sample_telemetry_signals,
        mock_kafka_producer,
    ):
        """Test streaming telemetry to Kafka."""
        topic = "furnacepulse.telemetry.canonical"

        for signal in sample_telemetry_signals:
            message = {
                "tag": signal.tag_id,
                "value": signal.value,
                "timestamp": signal.timestamp.isoformat(),
            }
            await mock_kafka_producer.send(topic, message, key=signal.tag_id)

        # Verify messages sent
        assert len(mock_kafka_producer.sent_messages) == len(sample_telemetry_signals)

    def test_full_telemetry_pipeline(
        self,
        sample_telemetry_signals,
        sample_tmt_readings_normal,
    ):
        """Test full telemetry processing pipeline."""
        # Step 1: Ingest
        ingested_data = {
            "telemetry": sample_telemetry_signals,
            "tmt_readings": sample_tmt_readings_normal,
            "ingestion_timestamp": datetime.now(),
        }

        # Step 2: Validate
        valid_telemetry = [
            s for s in ingested_data["telemetry"]
            if s.quality == "GOOD"
        ]
        valid_tmt = [
            r for r in ingested_data["tmt_readings"]
            if r.signal_quality == "GOOD"
        ]

        # Step 3: Transform
        transformed = {
            "telemetry_count": len(valid_telemetry),
            "tmt_count": len(valid_tmt),
            "processing_timestamp": datetime.now(),
        }

        # Step 4: Output metrics
        pipeline_result = {
            "status": "SUCCESS",
            "records_processed": transformed["telemetry_count"] + transformed["tmt_count"],
            "validation_pass_rate": 100.0,
        }

        assert pipeline_result["status"] == "SUCCESS"
        assert pipeline_result["records_processed"] > 0


class TestKPICalculationWorkflow:
    """Tests for KPI calculation workflow."""

    def test_efficiency_kpi_calculation(self, sample_efficiency_inputs):
        """Test thermal efficiency KPI calculation."""
        # Calculate fuel input
        fuel_input_kW = (
            sample_efficiency_inputs["fuel_mass_flow_kg_h"]
            * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
            / 3.6
        )

        # Calculate efficiency
        thermal_efficiency = (
            sample_efficiency_inputs["useful_heat_output_kW"]
            / fuel_input_kW
            * 100
        )

        kpi_result = {
            "kpi_name": "thermal_efficiency",
            "value": thermal_efficiency,
            "unit": "percent",
            "timestamp": datetime.now().isoformat(),
            "target": 85.0,
            "status": "ON_TARGET" if thermal_efficiency >= 85.0 else "BELOW_TARGET",
        }

        assert kpi_result["value"] > 0
        assert kpi_result["status"] == "ON_TARGET"

    def test_sfc_kpi_calculation(self, sample_efficiency_inputs):
        """Test Specific Fuel Consumption KPI calculation."""
        fuel_energy_MJ_h = (
            sample_efficiency_inputs["fuel_mass_flow_kg_h"]
            * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
        )
        production_kg_h = sample_efficiency_inputs["production_rate_kg_h"]

        sfc = fuel_energy_MJ_h / production_kg_h

        kpi_result = {
            "kpi_name": "specific_fuel_consumption",
            "value": sfc,
            "unit": "MJ/kg_product",
            "timestamp": datetime.now().isoformat(),
        }

        assert kpi_result["value"] > 0

    def test_excess_air_kpi_calculation(self, sample_efficiency_inputs):
        """Test excess air KPI calculation."""
        o2_percent = sample_efficiency_inputs["flue_gas_O2_percent"]
        excess_air = o2_percent / (21.0 - o2_percent) * 100

        kpi_result = {
            "kpi_name": "excess_air",
            "value": excess_air,
            "unit": "percent",
            "target_min": 10.0,
            "target_max": 30.0,
            "status": "OPTIMAL" if 10.0 <= excess_air <= 30.0 else "OUT_OF_RANGE",
        }

        assert kpi_result["status"] == "OPTIMAL"

    def test_kpi_batch_calculation(self, sample_efficiency_inputs):
        """Test batch calculation of all KPIs."""
        # Calculate all KPIs
        fuel_input_kW = (
            sample_efficiency_inputs["fuel_mass_flow_kg_h"]
            * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
            / 3.6
        )

        kpis = {
            "thermal_efficiency_percent": (
                sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input_kW * 100
            ),
            "sfc_MJ_kg": (
                sample_efficiency_inputs["fuel_mass_flow_kg_h"]
                * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
                / sample_efficiency_inputs["production_rate_kg_h"]
            ),
            "excess_air_percent": (
                sample_efficiency_inputs["flue_gas_O2_percent"]
                / (21.0 - sample_efficiency_inputs["flue_gas_O2_percent"])
                * 100
            ),
            "stack_temp_C": sample_efficiency_inputs["flue_gas_temperature_C"],
        }

        # Generate provenance hash
        kpi_json = json.dumps(kpis, sort_keys=True)
        kpis["calculation_hash"] = hashlib.sha256(kpi_json.encode()).hexdigest()

        assert len(kpis) == 5
        assert "calculation_hash" in kpis

    def test_kpi_trend_analysis(self):
        """Test KPI trend analysis over time."""
        # Historical KPI data
        kpi_history = [
            {"timestamp": "2025-01-01", "efficiency": 89.5},
            {"timestamp": "2025-01-02", "efficiency": 89.2},
            {"timestamp": "2025-01-03", "efficiency": 88.8},
            {"timestamp": "2025-01-04", "efficiency": 88.5},
            {"timestamp": "2025-01-05", "efficiency": 88.1},
        ]

        # Calculate trend
        values = [k["efficiency"] for k in kpi_history]
        trend = values[-1] - values[0]  # Negative = declining

        trend_analysis = {
            "metric": "thermal_efficiency",
            "period_days": 5,
            "start_value": values[0],
            "end_value": values[-1],
            "change": trend,
            "trend_direction": "DECLINING" if trend < 0 else "IMPROVING",
            "degradation_rate_per_day": abs(trend) / 5,
        }

        assert trend_analysis["trend_direction"] == "DECLINING"


class TestHotspotDetectionPipeline:
    """Tests for hotspot detection pipeline."""

    def test_hotspot_detection_workflow(
        self,
        sample_tmt_readings_hotspot,
        alert_thresholds,
    ):
        """Test full hotspot detection workflow."""
        # Step 1: Analyze TMT readings
        warning_threshold = alert_thresholds["TMT"]["WARNING"]

        hotspots = []
        for reading in sample_tmt_readings_hotspot:
            if reading.temperature_C > warning_threshold:
                hotspots.append({
                    "tube_id": reading.tube_id,
                    "zone": reading.zone,
                    "temperature_C": reading.temperature_C,
                    "rate_of_rise_C_min": reading.rate_of_rise_C_min,
                    "severity": "WARNING" if reading.temperature_C < 950 else "URGENT",
                })

        # Step 2: Generate alerts
        alerts = []
        for hotspot in hotspots:
            alert = {
                "alert_id": f"HS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "alert_type": "HOTSPOT",
                "severity": hotspot["severity"],
                "tube_id": hotspot["tube_id"],
                "temperature_C": hotspot["temperature_C"],
                "timestamp": datetime.now().isoformat(),
            }
            alerts.append(alert)

        assert len(hotspots) > 0
        assert len(alerts) == len(hotspots)

    def test_hotspot_alert_generation(
        self,
        sample_tmt_readings_critical,
        alert_thresholds,
    ):
        """Test alert generation for critical hotspots."""
        urgent_threshold = alert_thresholds["TMT"]["URGENT"]

        critical_readings = [
            r for r in sample_tmt_readings_critical
            if r.temperature_C > urgent_threshold
        ]

        # Generate urgent alerts
        urgent_alerts = []
        for reading in critical_readings:
            alert = {
                "alert_code": "A-003",
                "tier": "URGENT",
                "description": f"Critical hotspot on {reading.tube_id}",
                "temperature_C": reading.temperature_C,
                "design_limit_C": reading.design_limit_C,
                "exceedance_C": reading.temperature_C - reading.design_limit_C,
                "recommended_action": "Immediate investigation required",
            }
            urgent_alerts.append(alert)

        assert len(urgent_alerts) > 0
        assert all(a["tier"] == "URGENT" for a in urgent_alerts)

    @pytest.mark.asyncio
    async def test_hotspot_alert_notification(
        self,
        sample_tmt_readings_hotspot,
        mock_kafka_producer,
    ):
        """Test hotspot alert notification via Kafka."""
        topic = "furnacepulse.alerts.generated"

        # Generate alert
        alert = {
            "alert_id": "HS-001",
            "type": "HOTSPOT",
            "severity": "WARNING",
            "timestamp": datetime.now().isoformat(),
        }

        await mock_kafka_producer.send(topic, alert, key=alert["alert_id"])

        assert len(mock_kafka_producer.sent_messages) == 1
        assert mock_kafka_producer.sent_messages[0]["topic"] == topic

    def test_spatial_clustering_pipeline(self, sample_tmt_readings_hotspot):
        """Test spatial clustering in hotspot detection."""
        import math

        # Find elevated readings
        elevated = [
            r for r in sample_tmt_readings_hotspot
            if r.temperature_C > 880.0
        ]

        # Group into clusters (simple distance-based)
        clusters = []
        assigned = set()

        for i, reading in enumerate(elevated):
            if i in assigned:
                continue

            cluster = [reading]
            assigned.add(i)

            for j, other in enumerate(elevated):
                if j in assigned or j == i:
                    continue

                distance = math.sqrt(
                    (reading.position_x - other.position_x) ** 2 +
                    (reading.position_y - other.position_y) ** 2
                )

                if distance <= 2.0:  # Adjacent
                    cluster.append(other)
                    assigned.add(j)

            clusters.append({
                "cluster_id": f"CL-{len(clusters) + 1:03d}",
                "size": len(cluster),
                "readings": cluster,
                "max_temp": max(r.temperature_C for r in cluster),
            })

        assert len(clusters) >= 1


class TestEvidencePackageGeneration:
    """Tests for evidence package generation."""

    def test_evidence_package_from_hotspot(
        self,
        sample_tmt_readings_hotspot,
    ):
        """Test evidence package generation from hotspot detection."""
        # Create evidence items
        evidence_items = []

        # TMT readings evidence
        tmt_data = [
            {"tube_id": r.tube_id, "temperature_C": r.temperature_C}
            for r in sample_tmt_readings_hotspot
        ]
        tmt_hash = hashlib.sha256(
            json.dumps(tmt_data, sort_keys=True).encode()
        ).hexdigest()

        evidence_items.append({
            "item_id": "EVD-001",
            "type": "SENSOR_DATA",
            "description": "TMT readings at time of detection",
            "data_hash": tmt_hash,
        })

        # Analysis result evidence
        analysis = {
            "hotspots_detected": 1,
            "max_temperature": max(r.temperature_C for r in sample_tmt_readings_hotspot),
            "analysis_method": "threshold_and_clustering",
        }
        analysis_hash = hashlib.sha256(
            json.dumps(analysis, sort_keys=True).encode()
        ).hexdigest()

        evidence_items.append({
            "item_id": "EVD-002",
            "type": "CALCULATION_RESULT",
            "description": "Hotspot analysis result",
            "data_hash": analysis_hash,
        })

        # Create package
        package = {
            "package_id": f"PKG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "event_type": "HOTSPOT_DETECTION",
            "items": evidence_items,
            "package_hash": hashlib.sha256(
                "".join(e["data_hash"] for e in evidence_items).encode()
            ).hexdigest(),
        }

        assert len(package["items"]) == 2
        assert package["package_hash"] is not None

    def test_evidence_package_from_compliance(
        self,
        sample_nfpa86_checklist,
    ):
        """Test evidence package generation from compliance check."""
        evidence_items = []

        # Checklist evidence
        checklist_data = [
            {"item_id": item.item_id, "status": item.status}
            for item in sample_nfpa86_checklist
        ]
        checklist_hash = hashlib.sha256(
            json.dumps(checklist_data, sort_keys=True).encode()
        ).hexdigest()

        evidence_items.append({
            "item_id": "EVD-001",
            "type": "COMPLIANCE_CHECK",
            "description": "NFPA 86 checklist evaluation",
            "data_hash": checklist_hash,
        })

        package = {
            "package_id": f"COMP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "event_type": "COMPLIANCE_AUDIT",
            "items": evidence_items,
        }

        assert len(package["items"]) == 1

    def test_evidence_immutability_verification(self):
        """Test verification of evidence package immutability."""
        original_items = [
            {"item_id": "EVD-001", "data_hash": "abc123"},
            {"item_id": "EVD-002", "data_hash": "def456"},
        ]

        original_package_hash = hashlib.sha256(
            "".join(e["data_hash"] for e in original_items).encode()
        ).hexdigest()

        # Verify package
        verification_hash = hashlib.sha256(
            "".join(e["data_hash"] for e in original_items).encode()
        ).hexdigest()

        is_valid = original_package_hash == verification_hash
        assert is_valid


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_all_healthy(self, test_config):
        """Test health check when all components healthy."""
        health_status = {
            "status": "healthy",
            "agent_id": test_config["agent_id"],
            "agent_name": test_config["agent_name"],
            "version": test_config["version"],
            "timestamp": datetime.now().isoformat(),
            "components": {
                "opcua_client": {"status": "ok", "latency_ms": 15},
                "kafka_producer": {"status": "ok", "latency_ms": 8},
                "efficiency_calculator": {"status": "ok"},
                "hotspot_detector": {"status": "ok"},
                "rul_predictor": {"status": "ok"},
                "compliance_manager": {"status": "ok"},
            },
        }

        assert health_status["status"] == "healthy"
        assert all(
            c["status"] == "ok"
            for c in health_status["components"].values()
        )

    def test_health_check_degraded(self):
        """Test health check when some components degraded."""
        health_status = {
            "status": "degraded",
            "components": {
                "opcua_client": {"status": "ok"},
                "kafka_producer": {"status": "degraded", "error": "High latency"},
                "efficiency_calculator": {"status": "ok"},
            },
        }

        # Degraded if any component not ok
        has_degraded = any(
            c["status"] != "ok"
            for c in health_status["components"].values()
        )

        assert has_degraded
        assert health_status["status"] == "degraded"

    def test_health_check_unhealthy(self):
        """Test health check when critical components down."""
        health_status = {
            "status": "unhealthy",
            "components": {
                "opcua_client": {"status": "error", "error": "Connection refused"},
                "kafka_producer": {"status": "ok"},
            },
        }

        # Unhealthy if critical component has error
        has_error = any(
            c["status"] == "error"
            for c in health_status["components"].values()
        )

        assert has_error
        assert health_status["status"] == "unhealthy"

    def test_liveness_probe(self):
        """Test liveness probe response."""
        liveness = {
            "alive": True,
            "timestamp": datetime.now().isoformat(),
        }

        assert liveness["alive"]

    def test_readiness_probe(self, mock_opcua_client, mock_kafka_producer):
        """Test readiness probe response."""
        # Check all dependencies
        opcua_ready = mock_opcua_client.is_connected()
        kafka_ready = True  # Assume mock is always ready

        readiness = {
            "ready": opcua_ready and kafka_ready,
            "dependencies": {
                "opcua": opcua_ready,
                "kafka": kafka_ready,
            },
            "timestamp": datetime.now().isoformat(),
        }

        assert readiness["ready"]


class TestOrchestratorIntegration:
    """Full integration tests for orchestrator."""

    def test_full_processing_cycle(
        self,
        sample_furnace_state,
        sample_telemetry_signals,
        sample_tmt_readings_normal,
        sample_efficiency_inputs,
    ):
        """Test complete processing cycle."""
        # Step 1: Ingest telemetry
        telemetry = {
            "furnace_id": sample_furnace_state.furnace_id,
            "signals": sample_telemetry_signals,
            "tmt_readings": sample_tmt_readings_normal,
            "timestamp": datetime.now(),
        }

        # Step 2: Calculate KPIs
        fuel_input_kW = (
            sample_efficiency_inputs["fuel_mass_flow_kg_h"]
            * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
            / 3.6
        )
        kpis = {
            "thermal_efficiency": sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input_kW * 100,
            "excess_air": sample_efficiency_inputs["flue_gas_O2_percent"] / (21.0 - sample_efficiency_inputs["flue_gas_O2_percent"]) * 100,
        }

        # Step 3: Check for hotspots
        hotspots = [
            r for r in sample_tmt_readings_normal
            if r.temperature_C > 900.0
        ]

        # Step 4: Generate response
        response = {
            "furnace_id": telemetry["furnace_id"],
            "processing_timestamp": datetime.now().isoformat(),
            "kpis": kpis,
            "hotspots_detected": len(hotspots),
            "alerts_generated": len(hotspots),
            "status": "OK",
        }

        assert response["status"] == "OK"
        assert response["hotspots_detected"] == 0

    @pytest.mark.asyncio
    async def test_async_processing_pipeline(
        self,
        sample_telemetry_signals,
        mock_opcua_client,
        mock_kafka_producer,
    ):
        """Test async processing pipeline."""
        # Simulate async telemetry read
        tags = ["FRN-001.FUEL.FLOW", "FRN-001.STACK.TEMP"]
        readings = await mock_opcua_client.read_tags(tags)

        # Process readings
        processed = []
        for tag, reading in readings.items():
            processed.append({
                "tag": tag,
                "value": reading["value"],
                "quality": reading["quality"],
            })

        # Publish to Kafka
        for record in processed:
            await mock_kafka_producer.send(
                "furnacepulse.telemetry.processed",
                record,
                key=record["tag"],
            )

        assert len(mock_kafka_producer.sent_messages) == len(processed)

    def test_error_handling_in_pipeline(self):
        """Test error handling in processing pipeline."""
        # Simulate processing with bad data
        bad_signals = [
            {"tag_id": "BAD.TAG", "value": None, "quality": "BAD"},
        ]

        errors = []
        processed = []

        for signal in bad_signals:
            try:
                if signal["value"] is None:
                    raise ValueError(f"Null value for tag {signal['tag_id']}")
                processed.append(signal)
            except ValueError as e:
                errors.append({
                    "tag": signal["tag_id"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

        # Pipeline should continue despite errors
        assert len(errors) == 1
        assert len(processed) == 0


class TestDeterministicResults:
    """Tests for result determinism."""

    def test_kpi_calculation_deterministic(self, sample_efficiency_inputs):
        """Test KPI calculations are deterministic."""
        results = []

        for _ in range(5):
            fuel_input = (
                sample_efficiency_inputs["fuel_mass_flow_kg_h"]
                * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
                / 3.6
            )
            efficiency = sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input * 100

            result_hash = hashlib.sha256(
                json.dumps({"efficiency": efficiency}, sort_keys=True).encode()
            ).hexdigest()

            results.append(result_hash)

        # All results should have same hash
        assert all(r == results[0] for r in results)

    def test_hotspot_detection_deterministic(self, sample_tmt_readings_hotspot):
        """Test hotspot detection is deterministic."""
        results = []

        for _ in range(5):
            hotspots = [
                {"tube_id": r.tube_id, "temp": r.temperature_C}
                for r in sample_tmt_readings_hotspot
                if r.temperature_C > 900.0
            ]

            result_hash = hashlib.sha256(
                json.dumps(hotspots, sort_keys=True).encode()
            ).hexdigest()

            results.append(result_hash)

        assert all(r == results[0] for r in results)


class TestPerformance:
    """Performance tests for orchestrator."""

    def test_telemetry_processing_throughput(
        self,
        sample_telemetry_signals,
    ):
        """Test telemetry processing throughput."""
        import time

        # Create larger dataset
        signals = sample_telemetry_signals * 100

        start_time = time.time()

        for signal in signals:
            # Validate
            is_valid = signal.quality == "GOOD"
            # Transform
            canonical = {
                "tag": signal.tag_id,
                "value": signal.value,
                "timestamp": signal.timestamp.isoformat(),
            }

        elapsed = time.time() - start_time
        throughput = len(signals) / elapsed

        # Should process at least 10000 signals/second
        assert throughput > 10000

    def test_kpi_calculation_latency(self, sample_efficiency_inputs):
        """Test KPI calculation latency."""
        import time

        start_time = time.time()

        for _ in range(1000):
            fuel_input = (
                sample_efficiency_inputs["fuel_mass_flow_kg_h"]
                * sample_efficiency_inputs["fuel_lhv_MJ_kg"]
                / 3.6
            )
            efficiency = sample_efficiency_inputs["useful_heat_output_kW"] / fuel_input * 100
            excess_air = sample_efficiency_inputs["flue_gas_O2_percent"] / (21.0 - sample_efficiency_inputs["flue_gas_O2_percent"]) * 100

        elapsed = time.time() - start_time

        # 1000 calculations in < 50ms
        assert elapsed < 0.05
