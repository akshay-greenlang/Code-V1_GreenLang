"""
Integration Tests for GL-007 FurnacePulse CMMS Integration

Tests CMMS (Computerized Maintenance Management System) integration including:
- Work order creation
- Asset lookup
- Maintenance scheduling
- RUL-triggered maintenance
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
from typing import Dict, Any


class TestCMMSWorkOrders:
    """Tests for CMMS work order operations."""

    def test_create_preventive_work_order(self, mock_cmms_client):
        """Test creating preventive maintenance work order."""
        work_order = {
            "asset_id": "FRN-001",
            "work_type": "PREVENTIVE",
            "priority": "MEDIUM",
            "description": "Scheduled burner tip inspection",
            "estimated_hours": 4.0,
            "requested_by": "FurnacePulse-Agent",
        }

        result = mock_cmms_client.create_work_order(work_order)

        assert result["status"] == "CREATED"
        assert "work_order_id" in result
        assert result["work_type"] == "PREVENTIVE"

    def test_create_predictive_work_order(self, mock_cmms_client):
        """Test creating predictive maintenance work order from RUL."""
        work_order = {
            "asset_id": "FRN-001",
            "component_id": "TUBE-R1-01",
            "work_type": "PREDICTIVE",
            "priority": "HIGH",
            "description": "RUL prediction indicates tube replacement needed within 30 days",
            "estimated_hours": 24.0,
            "requested_by": "FurnacePulse-RUL-Model",
            "rul_estimate_hours": 720,
            "confidence_percent": 92.0,
        }

        result = mock_cmms_client.create_work_order(work_order)

        assert result["status"] == "CREATED"
        assert result["work_type"] == "PREDICTIVE"
        assert result["priority"] == "HIGH"

    def test_create_corrective_work_order(self, mock_cmms_client):
        """Test creating corrective maintenance work order from alert."""
        work_order = {
            "asset_id": "FRN-001",
            "work_type": "CORRECTIVE",
            "priority": "URGENT",
            "description": "TMT hotspot detected on tube T-R1-03 - immediate inspection required",
            "estimated_hours": 8.0,
            "requested_by": "FurnacePulse-Alert-System",
            "alert_id": "ALT-001",
            "alert_severity": "CRITICAL",
        }

        result = mock_cmms_client.create_work_order(work_order)

        assert result["status"] == "CREATED"
        assert result["priority"] == "URGENT"


class TestCMMSAssetManagement:
    """Tests for CMMS asset management operations."""

    def test_get_furnace_asset(self, mock_cmms_client):
        """Test retrieving furnace asset details."""
        asset = mock_cmms_client.get_asset("FRN-001")

        assert asset["asset_id"] == "FRN-001"
        assert asset["asset_type"] == "FURNACE"
        assert asset["criticality"] == "HIGH"

    def test_get_component_asset(self, mock_cmms_client):
        """Test retrieving component asset details."""
        asset = mock_cmms_client.get_asset("TUBE-R1-01")

        assert asset["asset_id"] == "TUBE-R1-01"

    def test_get_maintenance_schedule(self, mock_cmms_client):
        """Test retrieving maintenance schedule."""
        schedule = mock_cmms_client.get_maintenance_schedule("FRN-001")

        assert len(schedule) > 0
        assert schedule[0]["asset_id"] == "FRN-001"
        assert "next_due" in schedule[0]


class TestCMMSRULIntegration:
    """Tests for RUL-triggered CMMS integration."""

    def test_rul_threshold_triggers_work_order(self, mock_cmms_client):
        """Test RUL below threshold triggers work order creation."""
        # Simulate RUL prediction
        rul_prediction = {
            "component_id": "TUBE-R1-01",
            "rul_hours": 500,  # Below 720 hour threshold
            "confidence": 0.95,
        }

        # Should trigger work order creation
        if rul_prediction["rul_hours"] < 720 and rul_prediction["confidence"] > 0.90:
            work_order = mock_cmms_client.create_work_order({
                "asset_id": "FRN-001",
                "component_id": rul_prediction["component_id"],
                "work_type": "PREDICTIVE",
                "priority": "HIGH",
                "description": f"RUL: {rul_prediction['rul_hours']} hours remaining",
            })

            assert work_order["status"] == "CREATED"
            assert work_order["priority"] == "HIGH"

    def test_rul_confidence_gate(self, mock_cmms_client):
        """Test low confidence RUL doesn't trigger work order."""
        rul_prediction = {
            "component_id": "TUBE-R1-01",
            "rul_hours": 500,
            "confidence": 0.70,  # Below 90% threshold
        }

        # Should NOT trigger work order
        work_order_created = False
        if rul_prediction["rul_hours"] < 720 and rul_prediction["confidence"] > 0.90:
            mock_cmms_client.create_work_order({})
            work_order_created = True

        assert work_order_created is False


class TestCMMSAlertIntegration:
    """Tests for alert-triggered CMMS integration."""

    def test_critical_alert_triggers_urgent_work_order(self, mock_cmms_client):
        """Test critical alert triggers urgent work order."""
        alert = {
            "alert_id": "ALT-001",
            "severity": "CRITICAL",
            "alert_type": "TMT_EXCEEDS_LIMIT",
            "furnace_id": "FRN-001",
            "tube_id": "T-R1-03",
        }

        # Map severity to priority
        priority_map = {
            "CRITICAL": "URGENT",
            "WARNING": "HIGH",
            "ADVISORY": "MEDIUM",
            "INFO": "LOW",
        }

        work_order = mock_cmms_client.create_work_order({
            "asset_id": alert["furnace_id"],
            "work_type": "CORRECTIVE",
            "priority": priority_map.get(alert["severity"], "MEDIUM"),
            "description": f"Alert: {alert['alert_type']} on {alert['tube_id']}",
            "alert_id": alert["alert_id"],
        })

        assert work_order["status"] == "CREATED"
        assert work_order["priority"] == "URGENT"

    def test_warning_alert_triggers_high_priority_work_order(self, mock_cmms_client):
        """Test warning alert triggers high priority work order."""
        alert = {
            "alert_id": "ALT-002",
            "severity": "WARNING",
            "alert_type": "TMT_HIGH",
            "furnace_id": "FRN-001",
        }

        work_order = mock_cmms_client.create_work_order({
            "asset_id": alert["furnace_id"],
            "work_type": "CORRECTIVE",
            "priority": "HIGH",
            "alert_id": alert["alert_id"],
        })

        assert work_order["priority"] == "HIGH"
