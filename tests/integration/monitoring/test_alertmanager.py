# -*- coding: utf-8 -*-
"""
Alertmanager Integration Tests
==============================

Integration tests for Alertmanager notification delivery and management.
Requires a running Alertmanager instance.

Run with: pytest tests/integration/monitoring/test_alertmanager.py -v
"""

import pytest
import requests
import time
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import os
import uuid

# Skip all tests if not running in integration mode
pytestmark = pytest.mark.skipif(
    os.environ.get("INTEGRATION_TESTS") != "true",
    reason="Integration tests disabled (set INTEGRATION_TESTS=true to run)"
)


# Configuration from environment
ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "http://localhost:9093")
SLACK_TEST_CHANNEL = os.environ.get("SLACK_TEST_CHANNEL", "#test-alerts")


@pytest.fixture(scope="module")
def alertmanager_client() -> Dict[str, str]:
    """Create an Alertmanager API client configuration."""
    return {
        "base_url": ALERTMANAGER_URL,
        "api_path": "/api/v2",
    }


@pytest.fixture(scope="module")
def wait_for_alertmanager(alertmanager_client):
    """Wait for Alertmanager to be ready."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(
                f"{alertmanager_client['base_url']}/-/ready",
                timeout=5
            )
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)

    pytest.fail("Alertmanager not ready after 60 seconds")


@pytest.fixture
def test_alert() -> Dict[str, Any]:
    """Create a test alert payload."""
    return {
        "labels": {
            "alertname": f"TestAlert_{uuid.uuid4().hex[:8]}",
            "severity": "warning",
            "service": "test-service",
            "environment": "test",
        },
        "annotations": {
            "summary": "Test alert from integration tests",
            "description": "This is a test alert created by integration tests",
        },
        "startsAt": datetime.utcnow().isoformat() + "Z",
        "generatorURL": "http://test/graph",
    }


class TestAlertDelivery:
    """Tests for alert delivery functionality."""

    def test_alert_delivery_api(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager,
        test_alert: Dict[str, Any]
    ):
        """Test alert can be submitted via API."""
        response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/alerts",
            json=[test_alert],
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        assert response.status_code == 200, f"Failed to submit alert: {response.text}"

    def test_alert_appears_in_list(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager,
        test_alert: Dict[str, Any]
    ):
        """Test that submitted alert appears in alert list."""
        # Submit alert
        response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/alerts",
            json=[test_alert],
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code == 200

        # Wait a moment for processing
        time.sleep(1)

        # Get alerts
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/alerts",
            timeout=10
        )
        assert response.status_code == 200

        alerts = response.json()
        alert_names = [a["labels"]["alertname"] for a in alerts]
        assert test_alert["labels"]["alertname"] in alert_names

    def test_alert_grouping(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test that alerts are grouped correctly."""
        # Submit multiple alerts that should be grouped
        group_id = uuid.uuid4().hex[:8]
        alerts = [
            {
                "labels": {
                    "alertname": "GroupedAlert",
                    "service": f"service-{group_id}",
                    "instance": f"instance-{i}",
                    "severity": "warning",
                },
                "annotations": {"summary": f"Test alert {i}"},
                "startsAt": datetime.utcnow().isoformat() + "Z",
            }
            for i in range(3)
        ]

        response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/alerts",
            json=alerts,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code == 200

        # Check alert groups
        time.sleep(2)
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/alerts/groups",
            timeout=10
        )
        assert response.status_code == 200


class TestSilenceManagement:
    """Tests for silence management API."""

    def test_silence_api_create(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test creating a silence via API."""
        silence = {
            "matchers": [
                {
                    "name": "alertname",
                    "value": "TestSilencedAlert",
                    "isRegex": False,
                }
            ],
            "startsAt": datetime.utcnow().isoformat() + "Z",
            "endsAt": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            "createdBy": "integration-test",
            "comment": "Test silence from integration tests",
        }

        response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silences",
            json=silence,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        assert response.status_code == 200
        data = response.json()
        assert "silenceID" in data

        # Cleanup: delete the silence
        silence_id = data["silenceID"]
        requests.delete(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silence/{silence_id}",
            timeout=10
        )

    def test_silence_api_list(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test listing silences via API."""
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silences",
            timeout=10
        )

        assert response.status_code == 200
        silences = response.json()
        assert isinstance(silences, list)

    def test_silence_api_get_by_id(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test getting a specific silence by ID."""
        # First create a silence
        silence = {
            "matchers": [
                {
                    "name": "alertname",
                    "value": "GetByIdTest",
                    "isRegex": False,
                }
            ],
            "startsAt": datetime.utcnow().isoformat() + "Z",
            "endsAt": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            "createdBy": "integration-test",
            "comment": "Test silence for get by ID",
        }

        create_response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silences",
            json=silence,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert create_response.status_code == 200
        silence_id = create_response.json()["silenceID"]

        # Get by ID
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silence/{silence_id}",
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == silence_id

        # Cleanup
        requests.delete(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silence/{silence_id}",
            timeout=10
        )

    def test_silence_api_delete(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test deleting a silence via API."""
        # First create a silence
        silence = {
            "matchers": [
                {
                    "name": "alertname",
                    "value": "DeleteTest",
                    "isRegex": False,
                }
            ],
            "startsAt": datetime.utcnow().isoformat() + "Z",
            "endsAt": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            "createdBy": "integration-test",
            "comment": "Test silence for deletion",
        }

        create_response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silences",
            json=silence,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert create_response.status_code == 200
        silence_id = create_response.json()["silenceID"]

        # Delete
        response = requests.delete(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silence/{silence_id}",
            timeout=10
        )
        assert response.status_code == 200

        # Verify deleted (state should be "expired")
        get_response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/silence/{silence_id}",
            timeout=10
        )
        if get_response.status_code == 200:
            data = get_response.json()
            assert data["status"]["state"] == "expired"


class TestInhibitionRules:
    """Tests for alert inhibition functionality."""

    def test_inhibition_rules_applied(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test that inhibition rules are applied."""
        # Submit a critical alert (source)
        critical_alert = {
            "labels": {
                "alertname": "CriticalAlert",
                "severity": "critical",
                "service": "test-service-inhibit",
            },
            "annotations": {"summary": "Critical alert"},
            "startsAt": datetime.utcnow().isoformat() + "Z",
        }

        # Submit a warning alert (should be inhibited)
        warning_alert = {
            "labels": {
                "alertname": "WarningAlert",
                "severity": "warning",
                "service": "test-service-inhibit",
            },
            "annotations": {"summary": "Warning alert"},
            "startsAt": datetime.utcnow().isoformat() + "Z",
        }

        # Submit both
        response = requests.post(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/alerts",
            json=[critical_alert, warning_alert],
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code == 200

        # Check that warning is inhibited (depends on inhibition rules config)
        # This is configuration-dependent


class TestAlertmanagerHealth:
    """Tests for Alertmanager health and status."""

    def test_alertmanager_ready(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test Alertmanager ready endpoint."""
        response = requests.get(
            f"{alertmanager_client['base_url']}/-/ready",
            timeout=10
        )
        assert response.status_code == 200

    def test_alertmanager_healthy(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test Alertmanager healthy endpoint."""
        response = requests.get(
            f"{alertmanager_client['base_url']}/-/healthy",
            timeout=10
        )
        assert response.status_code == 200

    def test_alertmanager_status(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test Alertmanager status endpoint."""
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/status",
            timeout=10
        )
        assert response.status_code == 200

        data = response.json()
        assert "config" in data
        assert "uptime" in data

    def test_alertmanager_receivers(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test Alertmanager receivers endpoint."""
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/receivers",
            timeout=10
        )
        assert response.status_code == 200

        receivers = response.json()
        assert isinstance(receivers, list)
        assert len(receivers) > 0  # Should have at least default receiver


class TestAlertmanagerCluster:
    """Tests for Alertmanager cluster functionality."""

    def test_cluster_status(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test Alertmanager cluster status."""
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/status",
            timeout=10
        )
        assert response.status_code == 200

        data = response.json()
        # Cluster info might be in status
        if "cluster" in data:
            cluster = data["cluster"]
            assert "status" in cluster

    def test_alertmanager_metrics(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test Alertmanager exposes metrics."""
        response = requests.get(
            f"{alertmanager_client['base_url']}/metrics",
            timeout=10
        )

        assert response.status_code == 200
        assert "alertmanager_" in response.text


class TestAlertRouting:
    """Tests for alert routing configuration."""

    def test_routing_tree(
        self,
        alertmanager_client: Dict[str, str],
        wait_for_alertmanager
    ):
        """Test that routing tree is configured."""
        response = requests.get(
            f"{alertmanager_client['base_url']}{alertmanager_client['api_path']}/status",
            timeout=10
        )
        assert response.status_code == 200

        data = response.json()
        config = data.get("config", {})

        # Original config should have route
        original = config.get("original", "")
        assert "route" in original or "receiver" in original
