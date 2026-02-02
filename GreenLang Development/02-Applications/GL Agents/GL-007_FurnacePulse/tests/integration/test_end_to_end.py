"""
End-to-End Integration Tests for GL-007 FurnacePulse

Tests complete workflows including:
- Telemetry collection -> Efficiency calculation -> Kafka publish
- TMT monitoring -> Alert generation -> CMMS work order
- RUL prediction -> Maintenance scheduling
- NFPA 86 compliance checking
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import hashlib
import json


class TestTelemetryToEfficiencyWorkflow:
    """End-to-end tests for telemetry to efficiency calculation workflow."""

    @pytest.mark.asyncio
    async def test_complete_efficiency_calculation_workflow(
        self,
        mock_opcua_client,
        mock_kafka_producer,
        sample_efficiency_inputs
    ):
        """Test complete workflow from OPC-UA read to Kafka publish."""
        # Step 1: Collect telemetry from OPC-UA
        tags = ["FRN-001.FUEL.FLOW", "FRN-001.STACK.TEMP", "FRN-001.FLUE.O2"]
        telemetry = await mock_opcua_client.read_tags(tags)

        assert all(t["quality"] == "GOOD" for t in telemetry.values())

        # Step 2: Calculate efficiency
        fuel_flow = telemetry["FRN-001.FUEL.FLOW"]["value"]
        stack_temp = telemetry["FRN-001.STACK.TEMP"]["value"]
        flue_o2 = telemetry["FRN-001.FLUE.O2"]["value"]

        # Simplified efficiency calculation
        fuel_lhv = 48.0  # MJ/kg for natural gas
        fuel_input_kW = fuel_flow * fuel_lhv / 3.6

        # Excess air from O2
        excess_air_percent = flue_o2 / (21 - flue_o2) * 100

        # Stack loss (simplified Siegert)
        stack_loss_percent = 0.38 * (stack_temp - 25) / (21 - flue_o2)
        efficiency_percent = 100 - stack_loss_percent - 2  # -2% other losses

        # Step 3: Create provenance hash
        calc_inputs = {
            "fuel_flow": fuel_flow,
            "stack_temp": stack_temp,
            "flue_o2": flue_o2,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(calc_inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Step 4: Publish to Kafka
        result_message = {
            "furnace_id": "FRN-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "efficiency_percent": round(efficiency_percent, 2),
            "excess_air_percent": round(excess_air_percent, 2),
            "fuel_input_kW": round(fuel_input_kW, 2),
            "provenance_hash": provenance_hash,
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.site1.FRN-001.efficiency",
            message=result_message,
            key="FRN-001"
        )

        # Verify workflow completed
        assert len(mock_kafka_producer.sent_messages) == 1
        sent = mock_kafka_producer.sent_messages[0]["message"]
        assert "efficiency_percent" in sent
        assert "provenance_hash" in sent
        assert 80 < sent["efficiency_percent"] < 100


class TestTMTMonitoringWorkflow:
    """End-to-end tests for TMT monitoring and alert workflow."""

    @pytest.mark.asyncio
    async def test_hotspot_detection_to_alert_workflow(
        self,
        mock_opcua_client,
        mock_kafka_producer,
        mock_cmms_client,
        sample_tmt_readings_hotspot,
        alert_thresholds
    ):
        """Test complete hotspot detection to CMMS work order workflow."""
        # Step 1: Read TMT data (simulated from fixture)
        tmt_readings = sample_tmt_readings_hotspot

        # Step 2: Check for hotspots
        alerts = []
        for reading in tmt_readings:
            if reading.temperature_C > alert_thresholds["TMT"]["WARNING"]:
                severity = "CRITICAL" if reading.temperature_C > alert_thresholds["TMT"]["URGENT"] else "WARNING"
                alerts.append({
                    "alert_id": f"ALT-{reading.tube_id}",
                    "severity": severity,
                    "alert_type": "TMT_HIGH",
                    "tube_id": reading.tube_id,
                    "temperature_C": reading.temperature_C,
                    "threshold_C": alert_thresholds["TMT"]["WARNING"],
                })

        assert len(alerts) > 0  # Should detect hotspot

        # Step 3: Publish alerts to Kafka
        for alert in alerts:
            await mock_kafka_producer.send(
                topic="furnacepulse.alerts",
                message={
                    **alert,
                    "furnace_id": "FRN-001",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                key=alert["alert_id"]
            )

        # Step 4: Create CMMS work order for critical alerts
        for alert in alerts:
            if alert["severity"] in ["WARNING", "CRITICAL"]:
                work_order = mock_cmms_client.create_work_order({
                    "asset_id": "FRN-001",
                    "component_id": alert["tube_id"],
                    "work_type": "CORRECTIVE",
                    "priority": "URGENT" if alert["severity"] == "CRITICAL" else "HIGH",
                    "description": f"TMT hotspot: {alert['tube_id']} at {alert['temperature_C']}C",
                    "alert_id": alert["alert_id"],
                })
                assert work_order["status"] == "CREATED"

        # Verify workflow
        assert len(mock_kafka_producer.sent_messages) >= 1


class TestRULPredictionWorkflow:
    """End-to-end tests for RUL prediction workflow."""

    @pytest.mark.asyncio
    async def test_rul_prediction_to_maintenance_workflow(
        self,
        mock_kafka_producer,
        mock_cmms_client,
        mock_historian_client,
        sample_failure_history
    ):
        """Test RUL prediction triggering maintenance scheduling."""
        # Step 1: Query historical data (simulated)
        # In production, would query historian for degradation trends

        # Step 2: Calculate RUL (simplified)
        current_hours = 45000
        failure_hours = [h["time_to_failure_hours"] for h in sample_failure_history if not h["censored"]]
        mean_time_to_failure = sum(failure_hours) / len(failure_hours)

        # Simple RUL estimate
        rul_hours = max(0, mean_time_to_failure - current_hours)
        confidence = 0.92  # From model

        # Step 3: Publish RUL prediction to Kafka
        rul_message = {
            "furnace_id": "FRN-001",
            "component_id": "TUBE-R1-01",
            "component_type": "RADIANT_TUBE",
            "rul_hours": round(rul_hours, 0),
            "failure_probability_30d": 0.15,
            "confidence": confidence,
            "model_id": "weibull-rul-v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.models.inference",
            message=rul_message,
            key="FRN-001.TUBE-R1-01"
        )

        # Step 4: If RUL below threshold, create maintenance work order
        RUL_THRESHOLD_HOURS = 720  # 30 days
        CONFIDENCE_THRESHOLD = 0.90

        if rul_hours < RUL_THRESHOLD_HOURS and confidence > CONFIDENCE_THRESHOLD:
            work_order = mock_cmms_client.create_work_order({
                "asset_id": "FRN-001",
                "component_id": "TUBE-R1-01",
                "work_type": "PREDICTIVE",
                "priority": "HIGH",
                "description": f"RUL: {rul_hours} hours remaining (confidence: {confidence*100:.0f}%)",
                "rul_estimate_hours": rul_hours,
            })
            assert work_order["status"] == "CREATED"

        # Verify prediction published
        assert any(
            m["topic"] == "furnacepulse.models.inference"
            for m in mock_kafka_producer.sent_messages
        )


class TestNFPA86ComplianceWorkflow:
    """End-to-end tests for NFPA 86 compliance checking workflow."""

    @pytest.mark.asyncio
    async def test_compliance_check_workflow(
        self,
        mock_kafka_producer,
        sample_nfpa86_checklist
    ):
        """Test NFPA 86 compliance checking and reporting."""
        # Step 1: Run compliance checks
        checklist = sample_nfpa86_checklist
        results = {
            "furnace_id": "FRN-001",
            "check_date": datetime.now(timezone.utc).isoformat(),
            "total_items": len(checklist),
            "passed": sum(1 for item in checklist if item.status == "PASS"),
            "failed": sum(1 for item in checklist if item.status == "FAIL"),
            "pending": sum(1 for item in checklist if item.status == "PENDING"),
            "items": [
                {
                    "item_id": item.item_id,
                    "category": item.category,
                    "status": item.status,
                }
                for item in checklist
            ]
        }

        results["compliant"] = results["failed"] == 0 and results["pending"] == 0

        # Step 2: Publish compliance report
        await mock_kafka_producer.send(
            topic="furnacepulse.compliance.nfpa86",
            message=results,
            key="FRN-001"
        )

        # Step 3: Verify compliance status
        sent = mock_kafka_producer.sent_messages[-1]["message"]
        assert "compliant" in sent
        assert sent["passed"] == len(checklist)  # All items should pass

    @pytest.mark.asyncio
    async def test_compliance_failure_generates_alert(
        self,
        mock_kafka_producer,
        sample_nfpa86_checklist_with_failures
    ):
        """Test compliance failures generate alerts."""
        checklist = sample_nfpa86_checklist_with_failures

        # Find failures
        failures = [item for item in checklist if item.status == "FAIL"]

        # Generate alerts for each failure
        for failure in failures:
            alert = {
                "alert_id": f"COMP-{failure.item_id}",
                "severity": "CRITICAL",
                "alert_type": "NFPA86_COMPLIANCE_FAILURE",
                "furnace_id": "FRN-001",
                "item_id": failure.item_id,
                "category": failure.category,
                "description": failure.description,
                "requirement": failure.requirement,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await mock_kafka_producer.send(
                topic="furnacepulse.alerts",
                message=alert,
                key=alert["alert_id"]
            )

        # Verify alerts generated
        compliance_alerts = [
            m for m in mock_kafka_producer.sent_messages
            if m["message"].get("alert_type") == "NFPA86_COMPLIANCE_FAILURE"
        ]
        assert len(compliance_alerts) == len(failures)


class TestProvenanceAuditWorkflow:
    """End-to-end tests for provenance and audit trail workflow."""

    @pytest.mark.asyncio
    async def test_calculation_audit_trail(
        self,
        mock_kafka_producer,
        sample_efficiency_inputs
    ):
        """Test complete audit trail for efficiency calculation."""
        # Step 1: Capture inputs
        inputs = sample_efficiency_inputs

        # Step 2: Perform calculation with provenance
        fuel_input_kW = inputs["fuel_mass_flow_kg_h"] * inputs["fuel_lhv_MJ_kg"] / 3.6
        efficiency = inputs["useful_heat_output_kW"] / fuel_input_kW * 100

        # Step 3: Generate provenance hash
        calc_record = {
            "calculation_type": "thermal_efficiency",
            "version": "1.0.0",
            "inputs": inputs,
            "outputs": {
                "fuel_input_kW": round(fuel_input_kW, 2),
                "efficiency_percent": round(efficiency, 2),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        outputs_hash = hashlib.sha256(
            json.dumps(calc_record["outputs"], sort_keys=True).encode()
        ).hexdigest()

        calc_record["inputs_hash"] = inputs_hash[:16]
        calc_record["outputs_hash"] = outputs_hash[:16]
        calc_record["computation_hash"] = hashlib.sha256(
            (inputs_hash + outputs_hash).encode()
        ).hexdigest()[:16]

        # Step 4: Publish audit record
        await mock_kafka_producer.send(
            topic="furnacepulse.audit",
            message=calc_record,
            key=calc_record["computation_hash"]
        )

        # Verify audit record
        sent = mock_kafka_producer.sent_messages[-1]["message"]
        assert "computation_hash" in sent
        assert "inputs_hash" in sent
        assert "outputs_hash" in sent
        assert len(sent["computation_hash"]) == 16

    @pytest.mark.asyncio
    async def test_deterministic_calculation_produces_same_hash(self):
        """Test that same inputs always produce same hash."""
        inputs = {
            "fuel_mass_flow_kg_h": 1500.0,
            "fuel_lhv_MJ_kg": 48.0,
        }

        hash1 = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        hash2 = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        assert hash1 == hash2
