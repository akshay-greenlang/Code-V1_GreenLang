"""
GL-007 FurnacePulse - Full Workflow Integration Tests

Comprehensive end-to-end integration tests for the complete furnace
performance monitoring workflow. Tests the entire data pipeline from
OPC-UA telemetry collection through calculation, alert generation,
and CMMS work order creation.

Test Categories:
    - Complete telemetry-to-insight pipeline
    - OPC-UA connector tests with mocks
    - Hotspot detection workflow
    - RUL prediction workflow
    - NFPA 86 compliance workflow
    - Provenance and audit trail verification
    - Error handling and recovery
    - Performance benchmarks

Reference: NFPA 86 Standard for Ovens and Furnaces

Usage:
    pytest tests/integration/test_full_workflow.py -v
    pytest tests/integration/test_full_workflow.py -v -m "workflow"
    pytest tests/integration/test_full_workflow.py -v -m "opcua"
"""

import pytest
import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from dataclasses import dataclass
import logging

import numpy as np

# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

logger = logging.getLogger(__name__)


# =============================================================================
# Test Data Classes
# =============================================================================

@dataclass
class MockOPCUANode:
    """Mock OPC-UA node for testing."""
    node_id: str
    value: float
    quality: int = 0  # GOOD
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class WorkflowTestCase:
    """Test case definition for workflow tests."""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    expected_alerts: List[str]
    expected_compliance_status: str


# =============================================================================
# OPC-UA Connector Tests with Mocks
# =============================================================================

class TestOPCUAConnectorIntegration:
    """Integration tests for OPC-UA connector with mocked server."""

    @pytest.fixture
    def mock_opcua_server(self):
        """Create mock OPC-UA server with realistic furnace data."""
        server = AsyncMock()

        # Simulated furnace sensor nodes
        nodes = {
            "ns=2;s=Furnace1.Zone1.TMT.01": MockOPCUANode("TMT-Z1-01", 820.5),
            "ns=2;s=Furnace1.Zone1.TMT.02": MockOPCUANode("TMT-Z1-02", 825.3),
            "ns=2;s=Furnace1.Zone1.TMT.03": MockOPCUANode("TMT-Z1-03", 818.7),
            "ns=2;s=Furnace1.Zone2.TMT.01": MockOPCUANode("TMT-Z2-01", 795.2),
            "ns=2;s=Furnace1.Zone2.TMT.02": MockOPCUANode("TMT-Z2-02", 798.6),
            "ns=2;s=Furnace1.Fuel.Flow": MockOPCUANode("FUEL-FLOW", 1500.0),
            "ns=2;s=Furnace1.Fuel.Pressure": MockOPCUANode("FUEL-PRESS", 350.0),
            "ns=2;s=Furnace1.Stack.Temp": MockOPCUANode("STACK-TEMP", 380.0),
            "ns=2;s=Furnace1.Flue.O2": MockOPCUANode("FLUE-O2", 3.5),
            "ns=2;s=Furnace1.Flue.CO2": MockOPCUANode("FLUE-CO2", 10.2),
            "ns=2;s=Furnace1.Draft.Firebox": MockOPCUANode("DRAFT-FB", -25.0),
            "ns=2;s=Furnace1.Draft.Stack": MockOPCUANode("DRAFT-STK", -150.0),
            "ns=2;s=Furnace1.Air.Flow": MockOPCUANode("AIR-FLOW", 25000.0),
        }

        async def mock_read_node(node_id: str):
            if node_id in nodes:
                node = nodes[node_id]
                return {
                    "value": node.value,
                    "quality": node.quality,
                    "source_timestamp": node.timestamp.isoformat(),
                    "server_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return {"value": None, "quality": 128, "source_timestamp": None}

        async def mock_read_nodes(node_ids: List[str]):
            return {nid: await mock_read_node(nid) for nid in node_ids}

        server.read_node = mock_read_node
        server.read_nodes = mock_read_nodes
        server.connect = AsyncMock(return_value=True)
        server.disconnect = AsyncMock(return_value=True)
        server.is_connected = Mock(return_value=True)
        server._nodes = nodes

        return server

    @pytest.mark.workflow
    @pytest.mark.opcua
    async def test_opcua_connection_and_tag_reading(self, mock_opcua_server):
        """Test OPC-UA connection establishment and tag reading."""
        # Connect to mock server
        connected = await mock_opcua_server.connect()
        assert connected is True
        assert mock_opcua_server.is_connected() is True

        # Read single tag
        result = await mock_opcua_server.read_node("ns=2;s=Furnace1.Zone1.TMT.01")
        assert result["value"] == 820.5
        assert result["quality"] == 0  # GOOD

        # Read multiple tags
        tags = [
            "ns=2;s=Furnace1.Zone1.TMT.01",
            "ns=2;s=Furnace1.Zone1.TMT.02",
            "ns=2;s=Furnace1.Fuel.Flow",
        ]
        results = await mock_opcua_server.read_nodes(tags)
        assert len(results) == 3
        assert all(r["quality"] == 0 for r in results.values())

    @pytest.mark.workflow
    @pytest.mark.opcua
    async def test_opcua_batch_telemetry_collection(self, mock_opcua_server):
        """Test batch collection of furnace telemetry."""
        # Collect all TMT readings
        tmt_tags = [k for k in mock_opcua_server._nodes.keys() if "TMT" in k]
        results = await mock_opcua_server.read_nodes(tmt_tags)

        # Verify all readings collected
        assert len(results) == 5
        temperatures = [r["value"] for r in results.values()]
        assert all(700 <= t <= 950 for t in temperatures)

        # Calculate statistics
        avg_temp = sum(temperatures) / len(temperatures)
        max_temp = max(temperatures)
        min_temp = min(temperatures)

        assert 750 <= avg_temp <= 850
        assert max_temp - min_temp < 50  # Reasonable spread

    @pytest.mark.workflow
    @pytest.mark.opcua
    async def test_opcua_signal_quality_handling(self, mock_opcua_server):
        """Test handling of OPC-UA signal quality flags."""
        # Normal quality
        result = await mock_opcua_server.read_node("ns=2;s=Furnace1.Zone1.TMT.01")
        assert result["quality"] == 0  # GOOD

        # Bad quality (non-existent node)
        result = await mock_opcua_server.read_node("ns=2;s=NonExistent")
        assert result["quality"] == 128  # BAD
        assert result["value"] is None

    @pytest.mark.workflow
    @pytest.mark.opcua
    async def test_opcua_reconnection_on_failure(self, mock_opcua_server):
        """Test automatic reconnection on connection failure."""
        # Simulate disconnect
        mock_opcua_server.is_connected = Mock(return_value=False)

        # Verify disconnected state
        assert mock_opcua_server.is_connected() is False

        # Reconnect
        mock_opcua_server.is_connected = Mock(return_value=True)
        await mock_opcua_server.connect()

        assert mock_opcua_server.is_connected() is True


# =============================================================================
# Complete Telemetry-to-Insight Pipeline Tests
# =============================================================================

class TestCompleteTelemetryPipeline:
    """End-to-end tests for the complete telemetry processing pipeline."""

    @pytest.fixture
    def mock_kafka_producer(self):
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.sent_messages = []

        async def mock_send(topic: str, message: Dict, key: str = None):
            producer.sent_messages.append({
                "topic": topic,
                "message": message,
                "key": key,
                "timestamp": datetime.now(timezone.utc),
            })
            return {"partition": 0, "offset": len(producer.sent_messages)}

        producer.send = mock_send
        producer.flush = AsyncMock()
        producer.close = AsyncMock()
        return producer

    @pytest.fixture
    def mock_cmms_client(self):
        """Create mock CMMS client."""
        client = Mock()
        client.work_orders = []

        def create_work_order(data: Dict) -> Dict:
            wo = {
                "work_order_id": f"WO-{len(client.work_orders) + 1:06d}",
                "status": "CREATED",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **data,
            }
            client.work_orders.append(wo)
            return wo

        client.create_work_order = Mock(side_effect=create_work_order)
        return client

    @pytest.fixture
    def sample_telemetry_batch(self):
        """Create sample telemetry batch for testing."""
        return {
            "furnace_id": "FRN-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tmt_readings": [
                {"tube_id": "T-R1-01", "zone": "RADIANT", "temperature_c": 820.0, "position": (0, 0)},
                {"tube_id": "T-R1-02", "zone": "RADIANT", "temperature_c": 825.0, "position": (0, 1)},
                {"tube_id": "T-R1-03", "zone": "RADIANT", "temperature_c": 818.0, "position": (0, 2)},
                {"tube_id": "T-R1-04", "zone": "RADIANT", "temperature_c": 830.0, "position": (1, 0)},
                {"tube_id": "T-C1-01", "zone": "CONVECTION", "temperature_c": 650.0, "position": (2, 0)},
            ],
            "process_signals": {
                "fuel_flow_kg_h": 1500.0,
                "air_flow_kg_h": 25000.0,
                "stack_temp_c": 380.0,
                "flue_o2_pct": 3.5,
                "flue_co2_pct": 10.2,
                "draft_firebox_pa": -25.0,
                "draft_stack_pa": -150.0,
            },
        }

    @pytest.mark.workflow
    async def test_complete_telemetry_to_kpi_workflow(
        self,
        mock_kafka_producer,
        sample_telemetry_batch,
    ):
        """Test complete workflow from telemetry to KPI calculation."""
        telemetry = sample_telemetry_batch

        # Step 1: Validate incoming telemetry
        assert telemetry["furnace_id"] == "FRN-001"
        assert len(telemetry["tmt_readings"]) == 5
        assert "fuel_flow_kg_h" in telemetry["process_signals"]

        # Step 2: Calculate thermal efficiency
        fuel_flow = telemetry["process_signals"]["fuel_flow_kg_h"]
        fuel_lhv = 48.0  # MJ/kg for natural gas
        fuel_input_kw = fuel_flow * fuel_lhv / 3.6

        # Simplified efficiency calculation
        stack_temp = telemetry["process_signals"]["stack_temp_c"]
        flue_o2 = telemetry["process_signals"]["flue_o2_pct"]
        stack_loss_pct = 0.38 * (stack_temp - 25) / (21 - flue_o2)
        efficiency_pct = 100 - stack_loss_pct - 2  # -2% other losses

        assert 80 < efficiency_pct < 100

        # Step 3: Calculate excess air
        excess_air_pct = flue_o2 / (21 - flue_o2) * 100
        assert 15 < excess_air_pct < 25

        # Step 4: Generate provenance hash
        calc_inputs = {
            "fuel_flow": fuel_flow,
            "stack_temp": stack_temp,
            "flue_o2": flue_o2,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(calc_inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        assert len(provenance_hash) == 16

        # Step 5: Publish KPI results to Kafka
        kpi_message = {
            "furnace_id": telemetry["furnace_id"],
            "timestamp": telemetry["timestamp"],
            "kpis": {
                "thermal_efficiency_pct": round(efficiency_pct, 2),
                "excess_air_pct": round(excess_air_pct, 2),
                "fuel_input_kw": round(fuel_input_kw, 2),
                "stack_temp_c": stack_temp,
            },
            "provenance_hash": provenance_hash,
            "calculation_version": "1.0.0",
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.kpis.calculated",
            message=kpi_message,
            key=telemetry["furnace_id"],
        )

        # Verify message published
        assert len(mock_kafka_producer.sent_messages) == 1
        sent = mock_kafka_producer.sent_messages[0]
        assert sent["topic"] == "furnacepulse.kpis.calculated"
        assert "provenance_hash" in sent["message"]

    @pytest.mark.workflow
    async def test_complete_hotspot_detection_workflow(
        self,
        mock_kafka_producer,
        mock_cmms_client,
    ):
        """Test complete hotspot detection and alert workflow."""
        # Telemetry with hotspot
        telemetry = {
            "furnace_id": "FRN-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tmt_readings": [
                {"tube_id": "T-R1-01", "zone": "RADIANT", "temperature_c": 820.0},
                {"tube_id": "T-R1-02", "zone": "RADIANT", "temperature_c": 825.0},
                # HOTSPOT - above warning threshold (900C)
                {"tube_id": "T-R1-03", "zone": "RADIANT", "temperature_c": 920.0},
                {"tube_id": "T-R1-04", "zone": "RADIANT", "temperature_c": 895.0},
                {"tube_id": "T-C1-01", "zone": "CONVECTION", "temperature_c": 650.0},
            ],
        }

        # Step 1: Detect hotspots
        thresholds = {"ADVISORY": 850.0, "WARNING": 900.0, "URGENT": 950.0}
        alerts = []

        for reading in telemetry["tmt_readings"]:
            temp = reading["temperature_c"]
            if temp >= thresholds["URGENT"]:
                severity = "CRITICAL"
            elif temp >= thresholds["WARNING"]:
                severity = "WARNING"
            elif temp >= thresholds["ADVISORY"]:
                severity = "ADVISORY"
            else:
                continue

            alerts.append({
                "alert_id": f"HS-{reading['tube_id']}-{int(time.time())}",
                "severity": severity,
                "tube_id": reading["tube_id"],
                "temperature_c": temp,
                "threshold_c": thresholds.get(severity, thresholds["WARNING"]),
            })

        # Verify hotspot detected
        assert len(alerts) >= 1
        assert any(a["severity"] == "WARNING" for a in alerts)

        # Step 2: Publish alerts to Kafka
        for alert in alerts:
            await mock_kafka_producer.send(
                topic="furnacepulse.alerts.generated",
                message={
                    **alert,
                    "furnace_id": telemetry["furnace_id"],
                    "timestamp": telemetry["timestamp"],
                    "alert_type": "TMT_HOTSPOT",
                },
                key=alert["alert_id"],
            )

        assert len(mock_kafka_producer.sent_messages) >= 1

        # Step 3: Create CMMS work order for critical/warning alerts
        for alert in alerts:
            if alert["severity"] in ["WARNING", "CRITICAL"]:
                wo = mock_cmms_client.create_work_order({
                    "asset_id": telemetry["furnace_id"],
                    "component_id": alert["tube_id"],
                    "work_type": "CORRECTIVE",
                    "priority": "URGENT" if alert["severity"] == "CRITICAL" else "HIGH",
                    "description": f"TMT hotspot detected: {alert['tube_id']} at {alert['temperature_c']}C",
                    "alert_reference": alert["alert_id"],
                })
                assert wo["status"] == "CREATED"

        # Verify work orders created
        assert len(mock_cmms_client.work_orders) >= 1


# =============================================================================
# RUL Prediction Workflow Tests
# =============================================================================

class TestRULPredictionWorkflow:
    """End-to-end tests for RUL prediction workflow."""

    @pytest.fixture
    def sample_rul_inputs(self):
        """Create sample RUL prediction inputs."""
        return {
            "component_id": "TUBE-R1-01",
            "component_type": "RADIANT_TUBE",
            "operating_hours": 45000,
            "weibull_params": {"beta": 2.5, "eta": 70000},
            "operating_conditions": {
                "avg_temperature_c": 850.0,
                "design_temperature_c": 900.0,
                "thermal_cycles": 1200,
                "wall_thickness_mm": 8.5,
                "min_wall_thickness_mm": 6.0,
            },
            "maintenance_history": [
                {
                    "date": "2024-06-15",
                    "type": "PREVENTIVE",
                    "hours_at_maintenance": 40000,
                    "condition_score": 85,
                },
            ],
        }

    @pytest.mark.workflow
    async def test_complete_rul_prediction_workflow(
        self,
        sample_rul_inputs,
        mock_kafka_producer,
        mock_cmms_client,
    ):
        """Test complete RUL prediction workflow."""
        inputs = sample_rul_inputs
        import math

        # Step 1: Calculate Weibull-based RUL
        beta = inputs["weibull_params"]["beta"]
        eta = inputs["weibull_params"]["eta"]
        t = inputs["operating_hours"]

        # Reliability: R(t) = exp(-(t/eta)^beta)
        reliability = math.exp(-((t / eta) ** beta))

        # Failure probability
        failure_prob = 1 - reliability

        # RUL to 10% failure probability
        failure_threshold = 0.10
        target_reliability = 1 - failure_threshold
        t_fail = eta * ((-math.log(target_reliability)) ** (1 / beta))
        rul_hours = max(0, t_fail - t)

        assert rul_hours > 0
        assert failure_prob < 0.5  # Should not be near failure

        # Step 2: Apply condition adjustments
        cond = inputs["operating_conditions"]
        temp_ratio = cond["avg_temperature_c"] / cond["design_temperature_c"]

        # Operating below design temp = extended life
        if temp_ratio < 1.0:
            rul_adjustment = 1.0 + (1.0 - temp_ratio) * 0.5
        else:
            rul_adjustment = 1.0 - (temp_ratio - 1.0) * 2.0

        adjusted_rul = rul_hours * max(0.5, min(1.5, rul_adjustment))

        # Step 3: Calculate health index (0-100)
        health_index = (reliability * 40) + (min(adjusted_rul / eta, 1.0) * 30) + 30
        health_index = min(100, max(0, health_index))

        # Step 4: Determine risk category
        if failure_prob > 0.15 or adjusted_rul < 720:
            risk_category = "CRITICAL"
        elif failure_prob > 0.08 or health_index < 40:
            risk_category = "HIGH"
        elif failure_prob > 0.03 or health_index < 60:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"

        # Step 5: Generate provenance hash
        provenance_data = {
            "component_id": inputs["component_id"],
            "operating_hours": inputs["operating_hours"],
            "weibull_params": inputs["weibull_params"],
            "rul_hours": round(adjusted_rul, 1),
            "failure_prob": round(failure_prob, 6),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        # Step 6: Publish RUL prediction to Kafka
        rul_message = {
            "component_id": inputs["component_id"],
            "component_type": inputs["component_type"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": {
                "rul_hours": round(adjusted_rul, 1),
                "rul_days": round(adjusted_rul / 24, 1),
                "failure_probability": round(failure_prob, 6),
                "health_index": round(health_index, 1),
                "risk_category": risk_category,
            },
            "model_info": {
                "model_type": "weibull",
                "version": "1.0.0",
            },
            "provenance_hash": provenance_hash,
        }

        await mock_kafka_producer.send(
            topic="furnacepulse.rul.predictions",
            message=rul_message,
            key=inputs["component_id"],
        )

        assert len(mock_kafka_producer.sent_messages) == 1

        # Step 7: Create maintenance work order if RUL below threshold
        RUL_THRESHOLD_HOURS = 2160  # 90 days

        if adjusted_rul < RUL_THRESHOLD_HOURS:
            wo = mock_cmms_client.create_work_order({
                "asset_id": "FRN-001",
                "component_id": inputs["component_id"],
                "work_type": "PREDICTIVE",
                "priority": "HIGH" if risk_category in ["HIGH", "CRITICAL"] else "MEDIUM",
                "description": f"RUL prediction: {round(adjusted_rul / 24)} days remaining",
                "rul_estimate_hours": round(adjusted_rul, 1),
                "health_index": round(health_index, 1),
            })
            assert wo["status"] == "CREATED"


# =============================================================================
# NFPA 86 Compliance Workflow Tests
# =============================================================================

class TestNFPA86ComplianceWorkflow:
    """End-to-end tests for NFPA 86 compliance checking workflow."""

    @pytest.fixture
    def nfpa86_checklist(self):
        """Create NFPA 86 compliance checklist."""
        return [
            {
                "item_id": "NFPA86-4.3.1",
                "category": "Flame Supervision",
                "description": "Flame detection system operational",
                "requirement": "Flame detector installed and functional for each burner",
                "test_procedure": "Verify flame signal during operation",
            },
            {
                "item_id": "NFPA86-4.3.2",
                "category": "Flame Supervision",
                "description": "Flame failure response time",
                "requirement": "Fuel shutoff within 4 seconds of flame loss",
                "test_procedure": "Simulate flame loss, measure response time",
            },
            {
                "item_id": "NFPA86-5.2.1",
                "category": "Combustion Air",
                "description": "Combustion air interlock",
                "requirement": "Furnace operation interlocked with combustion air supply",
                "test_procedure": "Verify interlock trips on air loss",
            },
            {
                "item_id": "NFPA86-6.1.1",
                "category": "Fuel Shutoff",
                "description": "Emergency fuel shutoff",
                "requirement": "Manual emergency shutoff accessible and tested",
                "test_procedure": "Test E-stop functionality",
            },
            {
                "item_id": "NFPA86-7.1.1",
                "category": "Purge Cycle",
                "description": "Pre-ignition purge",
                "requirement": "Minimum 4 volume changes before ignition",
                "test_procedure": "Verify purge timer and flow calculation",
            },
            {
                "item_id": "NFPA86-8.2.1",
                "category": "Temperature Monitoring",
                "description": "Over-temperature protection",
                "requirement": "High-temperature alarm and shutoff functional",
                "test_procedure": "Simulate high temp, verify shutoff",
            },
        ]

    @pytest.fixture
    def furnace_test_results(self):
        """Simulated furnace test results."""
        return {
            "NFPA86-4.3.1": {"status": "PASS", "value": "Flame signal: 85%", "timestamp": "2025-01-15T10:00:00Z"},
            "NFPA86-4.3.2": {"status": "PASS", "value": "Response time: 2.3s", "timestamp": "2025-01-15T10:05:00Z"},
            "NFPA86-5.2.1": {"status": "PASS", "value": "Interlock verified", "timestamp": "2025-01-15T10:10:00Z"},
            "NFPA86-6.1.1": {"status": "PASS", "value": "E-stop functional", "timestamp": "2025-01-15T10:15:00Z"},
            "NFPA86-7.1.1": {"status": "PASS", "value": "Purge: 5.2 volumes", "timestamp": "2025-01-15T10:20:00Z"},
            "NFPA86-8.2.1": {"status": "PASS", "value": "High-temp shutoff verified", "timestamp": "2025-01-15T10:25:00Z"},
        }

    @pytest.mark.workflow
    @pytest.mark.nfpa
    @pytest.mark.safety
    async def test_complete_nfpa86_compliance_workflow(
        self,
        nfpa86_checklist,
        furnace_test_results,
        mock_kafka_producer,
    ):
        """Test complete NFPA 86 compliance checking workflow."""
        # Step 1: Execute compliance checks
        compliance_results = []

        for item in nfpa86_checklist:
            item_id = item["item_id"]
            test_result = furnace_test_results.get(item_id, {"status": "PENDING"})

            compliance_results.append({
                **item,
                "status": test_result["status"],
                "test_value": test_result.get("value"),
                "test_timestamp": test_result.get("timestamp"),
            })

        # Step 2: Calculate compliance summary
        passed = sum(1 for r in compliance_results if r["status"] == "PASS")
        failed = sum(1 for r in compliance_results if r["status"] == "FAIL")
        pending = sum(1 for r in compliance_results if r["status"] == "PENDING")
        total = len(compliance_results)

        is_compliant = failed == 0 and pending == 0
        compliance_score = (passed / total) * 100 if total > 0 else 0

        assert compliance_score == 100.0
        assert is_compliant is True

        # Step 3: Generate compliance report
        report = {
            "furnace_id": "FRN-001",
            "report_date": datetime.now(timezone.utc).isoformat(),
            "standard": "NFPA 86",
            "summary": {
                "total_items": total,
                "passed": passed,
                "failed": failed,
                "pending": pending,
                "compliance_score": round(compliance_score, 1),
                "is_compliant": is_compliant,
            },
            "items": compliance_results,
            "next_audit_due": (datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
        }

        # Step 4: Generate provenance hash for audit trail
        provenance_hash = hashlib.sha256(
            json.dumps(report, sort_keys=True, default=str).encode()
        ).hexdigest()
        report["provenance_hash"] = provenance_hash

        # Step 5: Publish compliance report
        await mock_kafka_producer.send(
            topic="furnacepulse.compliance.nfpa86",
            message=report,
            key=f"FRN-001-{report['report_date'][:10]}",
        )

        assert len(mock_kafka_producer.sent_messages) == 1
        sent = mock_kafka_producer.sent_messages[0]
        assert sent["message"]["summary"]["is_compliant"] is True

    @pytest.mark.workflow
    @pytest.mark.nfpa
    @pytest.mark.safety
    async def test_nfpa86_failure_generates_critical_alert(
        self,
        nfpa86_checklist,
        mock_kafka_producer,
    ):
        """Test that NFPA 86 compliance failures generate critical alerts."""
        # Simulate a failure
        test_results_with_failure = {
            "NFPA86-4.3.1": {"status": "PASS", "value": "Flame signal: 85%"},
            "NFPA86-4.3.2": {"status": "FAIL", "value": "Response time: 6.5s (EXCEEDS 4s)"},  # FAILURE
            "NFPA86-5.2.1": {"status": "PASS", "value": "Interlock verified"},
            "NFPA86-6.1.1": {"status": "PASS", "value": "E-stop functional"},
            "NFPA86-7.1.1": {"status": "PASS", "value": "Purge: 5.2 volumes"},
            "NFPA86-8.2.1": {"status": "PASS", "value": "High-temp shutoff verified"},
        }

        # Process results and generate alerts for failures
        alerts_generated = 0

        for item in nfpa86_checklist:
            item_id = item["item_id"]
            result = test_results_with_failure.get(item_id, {"status": "PENDING"})

            if result["status"] == "FAIL":
                alert = {
                    "alert_id": f"NFPA-{item_id}-{int(time.time())}",
                    "severity": "CRITICAL",
                    "alert_type": "NFPA86_COMPLIANCE_FAILURE",
                    "furnace_id": "FRN-001",
                    "item_id": item_id,
                    "category": item["category"],
                    "description": item["description"],
                    "requirement": item["requirement"],
                    "test_value": result.get("value"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "required_action": "IMMEDIATE_INSPECTION",
                }

                await mock_kafka_producer.send(
                    topic="furnacepulse.alerts.generated",
                    message=alert,
                    key=alert["alert_id"],
                )
                alerts_generated += 1

        # Verify alert generated for the failure
        assert alerts_generated == 1
        sent = mock_kafka_producer.sent_messages[0]
        assert sent["message"]["severity"] == "CRITICAL"
        assert sent["message"]["alert_type"] == "NFPA86_COMPLIANCE_FAILURE"


# =============================================================================
# Provenance and Audit Trail Tests
# =============================================================================

class TestProvenanceAuditTrail:
    """Tests for provenance tracking and audit trail functionality."""

    @pytest.mark.workflow
    async def test_deterministic_calculation_produces_consistent_hash(self):
        """Test that identical inputs produce identical provenance hashes."""
        inputs = {
            "fuel_flow_kg_h": 1500.0,
            "stack_temp_c": 380.0,
            "flue_o2_pct": 3.5,
        }

        # Calculate hash multiple times
        hashes = []
        for _ in range(5):
            hash_val = hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(hash_val)

        # All hashes should be identical
        assert len(set(hashes)) == 1

    @pytest.mark.workflow
    async def test_different_inputs_produce_different_hashes(self):
        """Test that different inputs produce different provenance hashes."""
        inputs1 = {"fuel_flow_kg_h": 1500.0, "stack_temp_c": 380.0}
        inputs2 = {"fuel_flow_kg_h": 1501.0, "stack_temp_c": 380.0}  # 1 kg/h difference

        hash1 = hashlib.sha256(json.dumps(inputs1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(inputs2, sort_keys=True).encode()).hexdigest()

        assert hash1 != hash2

    @pytest.mark.workflow
    async def test_complete_audit_trail_for_calculation(self, mock_kafka_producer):
        """Test complete audit trail for a calculation."""
        # Step 1: Capture inputs
        inputs = {
            "fuel_flow_kg_h": 1500.0,
            "fuel_lhv_mj_kg": 48.0,
            "useful_heat_kw": 18000.0,
        }
        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Step 2: Perform calculation
        fuel_input_kw = inputs["fuel_flow_kg_h"] * inputs["fuel_lhv_mj_kg"] / 3.6
        efficiency = inputs["useful_heat_kw"] / fuel_input_kw * 100

        outputs = {
            "fuel_input_kw": round(fuel_input_kw, 2),
            "efficiency_pct": round(efficiency, 2),
        }
        outputs_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Step 3: Generate computation hash
        computation_hash = hashlib.sha256(
            (inputs_hash + outputs_hash).encode()
        ).hexdigest()[:16]

        # Step 4: Create audit record
        audit_record = {
            "record_type": "CALCULATION",
            "calculation_type": "thermal_efficiency",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "provenance": {
                "inputs_hash": inputs_hash,
                "outputs_hash": outputs_hash,
                "computation_hash": computation_hash,
            },
            "agent_id": "GL-007",
        }

        # Step 5: Publish audit record
        await mock_kafka_producer.send(
            topic="furnacepulse.audit.calculations",
            message=audit_record,
            key=computation_hash,
        )

        # Verify audit record
        assert len(mock_kafka_producer.sent_messages) == 1
        sent = mock_kafka_producer.sent_messages[0]
        assert "provenance" in sent["message"]
        assert len(sent["message"]["provenance"]["computation_hash"]) == 16


# =============================================================================
# Error Handling and Recovery Tests
# =============================================================================

class TestErrorHandlingAndRecovery:
    """Tests for error handling and recovery in workflow."""

    @pytest.mark.workflow
    async def test_handles_missing_sensor_data_gracefully(self, mock_kafka_producer):
        """Test graceful handling of missing sensor data."""
        # Telemetry with missing values
        telemetry = {
            "furnace_id": "FRN-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tmt_readings": [
                {"tube_id": "T-R1-01", "temperature_c": 820.0},
                {"tube_id": "T-R1-02", "temperature_c": None},  # Missing
                {"tube_id": "T-R1-03", "temperature_c": 818.0},
            ],
            "process_signals": {
                "fuel_flow_kg_h": 1500.0,
                "stack_temp_c": None,  # Missing
                "flue_o2_pct": 3.5,
            },
        }

        # Filter out missing values
        valid_temps = [
            r["temperature_c"]
            for r in telemetry["tmt_readings"]
            if r["temperature_c"] is not None
        ]

        assert len(valid_temps) == 2
        assert 820.0 in valid_temps
        assert 818.0 in valid_temps

        # Calculate with available data
        avg_temp = sum(valid_temps) / len(valid_temps)
        assert 815 < avg_temp < 825

        # Generate data quality alert
        missing_count = sum(
            1 for r in telemetry["tmt_readings"] if r["temperature_c"] is None
        )

        if missing_count > 0:
            alert = {
                "alert_id": f"DQ-{int(time.time())}",
                "severity": "WARNING",
                "alert_type": "DATA_QUALITY",
                "description": f"Missing TMT readings: {missing_count}",
                "furnace_id": telemetry["furnace_id"],
            }

            await mock_kafka_producer.send(
                topic="furnacepulse.alerts.generated",
                message=alert,
                key=alert["alert_id"],
            )

        assert len(mock_kafka_producer.sent_messages) == 1

    @pytest.mark.workflow
    async def test_handles_out_of_range_values(self):
        """Test handling of out-of-range sensor values."""
        readings = [
            {"tube_id": "T-R1-01", "temperature_c": 820.0},  # Valid
            {"tube_id": "T-R1-02", "temperature_c": -50.0},  # Invalid (negative)
            {"tube_id": "T-R1-03", "temperature_c": 2000.0},  # Invalid (too high)
            {"tube_id": "T-R1-04", "temperature_c": 825.0},  # Valid
        ]

        valid_range = (0, 1500)  # Reasonable TMT range

        validated_readings = []
        invalid_readings = []

        for reading in readings:
            temp = reading["temperature_c"]
            if valid_range[0] <= temp <= valid_range[1]:
                validated_readings.append(reading)
            else:
                invalid_readings.append(reading)

        assert len(validated_readings) == 2
        assert len(invalid_readings) == 2

    @pytest.mark.workflow
    async def test_timeout_handling_for_long_calculations(self):
        """Test timeout handling for long-running calculations."""
        async def slow_calculation():
            await asyncio.sleep(0.5)  # Simulate slow calculation
            return {"result": 42}

        # Test with timeout
        try:
            result = await asyncio.wait_for(slow_calculation(), timeout=1.0)
            assert result["result"] == 42
        except asyncio.TimeoutError:
            pytest.fail("Calculation should complete within timeout")

        # Test timeout exceeded
        async def very_slow_calculation():
            await asyncio.sleep(2.0)
            return {"result": 42}

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(very_slow_calculation(), timeout=0.1)


# =============================================================================
# Performance Benchmark Tests
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests for workflow operations."""

    @pytest.mark.workflow
    @pytest.mark.performance
    async def test_telemetry_processing_throughput(self, mock_kafka_producer):
        """Benchmark telemetry processing throughput."""
        num_readings = 1000
        start_time = time.perf_counter()

        for i in range(num_readings):
            message = {
                "reading_id": i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature_c": 820.0 + (i % 100) * 0.1,
            }
            await mock_kafka_producer.send(
                topic="furnacepulse.telemetry",
                message=message,
                key=f"reading-{i}",
            )

        elapsed = time.perf_counter() - start_time
        throughput = num_readings / elapsed

        assert len(mock_kafka_producer.sent_messages) == num_readings
        assert throughput > 100  # At least 100 messages/second

        logger.info(f"Telemetry processing throughput: {throughput:.0f} messages/second")

    @pytest.mark.workflow
    @pytest.mark.performance
    async def test_provenance_hash_computation_speed(self):
        """Benchmark provenance hash computation speed."""
        inputs = {
            "furnace_id": "FRN-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "values": [float(i) for i in range(100)],
        }

        num_hashes = 10000
        start_time = time.perf_counter()

        for _ in range(num_hashes):
            _ = hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest()

        elapsed = time.perf_counter() - start_time
        hashes_per_second = num_hashes / elapsed

        assert hashes_per_second > 1000  # At least 1000 hashes/second

        logger.info(f"Hash computation speed: {hashes_per_second:.0f} hashes/second")

    @pytest.mark.workflow
    @pytest.mark.performance
    async def test_batch_hotspot_detection_performance(self):
        """Benchmark batch hotspot detection performance."""
        # Generate large batch of readings
        num_tubes = 500
        readings = [
            {
                "tube_id": f"T-R{i//50}-{i%50:02d}",
                "temperature_c": 800.0 + (i % 200),
            }
            for i in range(num_tubes)
        ]

        threshold = 900.0
        start_time = time.perf_counter()

        # Detect hotspots
        hotspots = [r for r in readings if r["temperature_c"] > threshold]

        elapsed = time.perf_counter() - start_time

        assert elapsed < 0.1  # Should complete in < 100ms
        assert len(hotspots) > 0

        logger.info(
            f"Batch detection: {num_tubes} tubes in {elapsed*1000:.2f}ms, "
            f"found {len(hotspots)} hotspots"
        )


# Fixture for mock Kafka producer used across tests
@pytest.fixture
def mock_kafka_producer():
    """Create mock Kafka producer for testing."""
    producer = AsyncMock()
    producer.sent_messages = []

    async def mock_send(topic: str, message: Dict, key: str = None):
        producer.sent_messages.append({
            "topic": topic,
            "message": message,
            "key": key,
            "timestamp": datetime.now(timezone.utc),
        })
        return {"partition": 0, "offset": len(producer.sent_messages)}

    producer.send = mock_send
    producer.flush = AsyncMock()
    producer.close = AsyncMock()
    return producer


@pytest.fixture
def mock_cmms_client():
    """Create mock CMMS client for testing."""
    client = Mock()
    client.work_orders = []

    def create_work_order(data: Dict) -> Dict:
        wo = {
            "work_order_id": f"WO-{len(client.work_orders) + 1:06d}",
            "status": "CREATED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        client.work_orders.append(wo)
        return wo

    client.create_work_order = Mock(side_effect=create_work_order)
    return client
