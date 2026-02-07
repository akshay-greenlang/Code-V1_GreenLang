"""
Unit tests for waf_routes API.

Tests all WAF management API endpoints including rule CRUD,
traffic analysis, anomaly detection, and mitigation management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.waf_management.api.waf_routes import router
from greenlang.infrastructure.waf_management.models import (
    WAFRule,
    RuleType,
    RuleAction,
    RulePriority,
    TrafficBaseline,
    DetectionResult,
    DetectionType,
    Severity,
)


def _build_test_app() -> FastAPI:
    """Build FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/waf")
    return app


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    app = _build_test_app()
    return TestClient(app)


class TestRuleListEndpoint:
    """Test GET /api/v1/waf/rules endpoint."""

    def test_list_rules_returns_200(self, test_client, waf_viewer_headers):
        """Test listing rules returns 200."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_rules.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/rules",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_list_rules_returns_rules(
        self, test_client, waf_viewer_headers, sample_rate_limit_rule
    ):
        """Test listing rules returns rule data."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_rules.return_value = [sample_rate_limit_rule]
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/rules",
                headers=waf_viewer_headers,
            )

            data = response.json()
            assert len(data["rules"]) == 1
            assert data["rules"][0]["name"] == sample_rate_limit_rule.name

    def test_list_rules_filter_by_type(self, test_client, waf_viewer_headers):
        """Test filtering rules by type."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_rules.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/rules?rule_type=rate_limit",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_list_rules_filter_by_enabled(self, test_client, waf_viewer_headers):
        """Test filtering rules by enabled status."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_rules.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/rules?enabled=true",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_list_rules_unauthorized(self, test_client):
        """Test listing rules without auth returns 401."""
        response = test_client.get("/api/v1/waf/rules")

        assert response.status_code in [401, 403]


class TestRuleCreateEndpoint:
    """Test POST /api/v1/waf/rules endpoint."""

    def test_create_rule_returns_201(self, test_client, waf_admin_headers):
        """Test creating rule returns 201."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_rule.return_value = WAFRule(
                rule_id=str(uuid4()),
                name="New Rate Limit",
                description="Test rule",
                rule_type=RuleType.RATE_LIMIT,
                action=RuleAction.BLOCK,
                priority=RulePriority.MEDIUM,
                conditions={},
                parameters={"limit": 100, "window_seconds": 60},
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="waf-admin",
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/waf/rules",
                headers=waf_admin_headers,
                json={
                    "name": "New Rate Limit",
                    "description": "Test rule",
                    "rule_type": "rate_limit",
                    "action": "block",
                    "parameters": {"limit": 100, "window_seconds": 60},
                },
            )

            assert response.status_code == 201

    def test_create_sqli_rule(self, test_client, waf_admin_headers):
        """Test creating SQL injection rule."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_rule.return_value = WAFRule(
                rule_id=str(uuid4()),
                name="SQLi Detection",
                description="Detect SQL injection",
                rule_type=RuleType.SQL_INJECTION,
                action=RuleAction.BLOCK,
                priority=RulePriority.CRITICAL,
                conditions={},
                parameters={"sensitivity": "high"},
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="waf-admin",
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/waf/rules",
                headers=waf_admin_headers,
                json={
                    "name": "SQLi Detection",
                    "description": "Detect SQL injection",
                    "rule_type": "sql_injection",
                    "action": "block",
                    "parameters": {"sensitivity": "high"},
                },
            )

            assert response.status_code == 201

    def test_create_rule_validation_error(self, test_client, waf_admin_headers):
        """Test creating rule with invalid data returns 422."""
        response = test_client.post(
            "/api/v1/waf/rules",
            headers=waf_admin_headers,
            json={
                "name": "",  # Empty name
                "rule_type": "invalid_type",
            },
        )

        assert response.status_code == 422

    def test_create_rule_viewer_forbidden(self, test_client, waf_viewer_headers):
        """Test creating rule as viewer returns 403."""
        response = test_client.post(
            "/api/v1/waf/rules",
            headers=waf_viewer_headers,
            json={
                "name": "Test Rule",
                "description": "Test",
                "rule_type": "rate_limit",
            },
        )

        assert response.status_code == 403


class TestRuleGetEndpoint:
    """Test GET /api/v1/waf/rules/{rule_id} endpoint."""

    def test_get_rule_returns_200(
        self, test_client, waf_viewer_headers, sample_rate_limit_rule
    ):
        """Test getting rule returns 200."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_rule.return_value = sample_rate_limit_rule
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/waf/rules/{sample_rate_limit_rule.rule_id}",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["rule_id"] == sample_rate_limit_rule.rule_id

    def test_get_rule_not_found(self, test_client, waf_viewer_headers):
        """Test getting nonexistent rule returns 404."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_rule.return_value = None
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/rules/nonexistent-id",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 404


class TestRuleUpdateEndpoint:
    """Test PUT /api/v1/waf/rules/{rule_id} endpoint."""

    def test_update_rule_returns_200(
        self, test_client, waf_admin_headers, sample_rate_limit_rule
    ):
        """Test updating rule returns 200."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_rule.return_value = sample_rate_limit_rule
            mock_service.update_rule.return_value = sample_rate_limit_rule
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/waf/rules/{sample_rate_limit_rule.rule_id}",
                headers=waf_admin_headers,
                json={
                    "description": "Updated description",
                    "parameters": {"limit": 200},
                },
            )

            assert response.status_code == 200

    def test_enable_rule(self, test_client, waf_admin_headers, sample_rate_limit_rule):
        """Test enabling a rule."""
        sample_rate_limit_rule.enabled = False

        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_rule.return_value = sample_rate_limit_rule
            sample_rate_limit_rule.enabled = True
            mock_service.update_rule.return_value = sample_rate_limit_rule
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/waf/rules/{sample_rate_limit_rule.rule_id}",
                headers=waf_admin_headers,
                json={"enabled": True},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True


class TestRuleDeleteEndpoint:
    """Test DELETE /api/v1/waf/rules/{rule_id} endpoint."""

    def test_delete_rule_returns_204(
        self, test_client, waf_admin_headers, sample_rate_limit_rule
    ):
        """Test deleting rule returns 204."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_rule.return_value = sample_rate_limit_rule
            mock_service.delete_rule.return_value = True
            mock_get_service.return_value = mock_service

            response = test_client.delete(
                f"/api/v1/waf/rules/{sample_rate_limit_rule.rule_id}",
                headers=waf_admin_headers,
            )

            assert response.status_code == 204

    def test_delete_rule_viewer_forbidden(
        self, test_client, waf_viewer_headers, sample_rate_limit_rule
    ):
        """Test deleting rule as viewer returns 403."""
        response = test_client.delete(
            f"/api/v1/waf/rules/{sample_rate_limit_rule.rule_id}",
            headers=waf_viewer_headers,
        )

        assert response.status_code == 403


class TestTrafficAnalysisEndpoints:
    """Test traffic analysis API endpoints."""

    def test_analyze_traffic_returns_200(self, test_client, waf_admin_headers):
        """Test analyzing traffic returns 200."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_anomaly_detector"
        ) as mock_get_detector:
            mock_detector = AsyncMock()
            mock_detector.analyze_traffic.return_value = []
            mock_get_detector.return_value = mock_detector

            response = test_client.post(
                "/api/v1/waf/traffic/analyze",
                headers=waf_admin_headers,
                json={
                    "start_time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                },
            )

            assert response.status_code == 200

    def test_get_traffic_summary(self, test_client, waf_viewer_headers):
        """Test getting traffic summary."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_traffic_summary.return_value = {
                "total_requests": 1000000,
                "blocked_requests": 5000,
                "requests_by_country": {"US": 500000, "GB": 200000},
            }
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/traffic/summary?period=24h",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "total_requests" in data


class TestBaselineEndpoints:
    """Test traffic baseline API endpoints."""

    def test_list_baselines(self, test_client, waf_viewer_headers):
        """Test listing traffic baselines."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_anomaly_detector"
        ) as mock_get_detector:
            mock_detector = AsyncMock()
            mock_detector.list_baselines.return_value = []
            mock_get_detector.return_value = mock_detector

            response = test_client.get(
                "/api/v1/waf/baselines",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_create_baseline(
        self, test_client, waf_admin_headers, sample_traffic_baseline
    ):
        """Test creating traffic baseline."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_anomaly_detector"
        ) as mock_get_detector:
            mock_detector = AsyncMock()
            mock_detector.build_baseline.return_value = sample_traffic_baseline
            mock_get_detector.return_value = mock_detector

            response = test_client.post(
                "/api/v1/waf/baselines",
                headers=waf_admin_headers,
                json={
                    "name": "API Baseline",
                    "endpoint_pattern": "/api/.*",
                    "time_window_hours": 24,
                },
            )

            assert response.status_code == 201

    def test_get_baseline(
        self, test_client, waf_viewer_headers, sample_traffic_baseline
    ):
        """Test getting traffic baseline."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_anomaly_detector"
        ) as mock_get_detector:
            mock_detector = AsyncMock()
            mock_detector.get_baseline.return_value = sample_traffic_baseline
            mock_get_detector.return_value = mock_detector

            response = test_client.get(
                f"/api/v1/waf/baselines/{sample_traffic_baseline.baseline_id}",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200


class TestDetectionEndpoints:
    """Test anomaly detection API endpoints."""

    def test_list_detections(self, test_client, waf_viewer_headers):
        """Test listing detections."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_detections.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/detections",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_list_detections_filter_by_type(self, test_client, waf_viewer_headers):
        """Test filtering detections by type."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_detections.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/detections?detection_type=sql_injection",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_list_detections_filter_by_severity(self, test_client, waf_viewer_headers):
        """Test filtering detections by severity."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_detections.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/detections?severity=critical",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200

    def test_get_detection_details(
        self, test_client, waf_viewer_headers, sqli_detection_result
    ):
        """Test getting detection details."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_detection.return_value = sqli_detection_result
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/waf/detections/{sqli_detection_result.detection_id}",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200


class TestMitigationEndpoints:
    """Test mitigation action endpoints."""

    def test_block_ip(self, test_client, waf_admin_headers):
        """Test blocking an IP address."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.block_ip.return_value = True
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/waf/mitigations/block-ip",
                headers=waf_admin_headers,
                json={
                    "ip_address": "45.33.32.156",
                    "duration_hours": 24,
                    "reason": "SQL injection attempt",
                },
            )

            assert response.status_code == 200

    def test_unblock_ip(self, test_client, waf_admin_headers):
        """Test unblocking an IP address."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.unblock_ip.return_value = True
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/waf/mitigations/unblock-ip",
                headers=waf_admin_headers,
                json={
                    "ip_address": "45.33.32.156",
                },
            )

            assert response.status_code == 200

    def test_list_blocked_ips(self, test_client, waf_viewer_headers):
        """Test listing blocked IPs."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_blocked_ips.return_value = [
                {
                    "ip_address": "45.33.32.156",
                    "blocked_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                    "reason": "SQL injection attempt",
                }
            ]
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/mitigations/blocked-ips",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["blocked_ips"]) >= 1


class TestMetricsEndpoints:
    """Test WAF metrics endpoints."""

    def test_get_waf_metrics(self, test_client, waf_viewer_headers):
        """Test getting WAF metrics."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_metrics.return_value = {
                "total_requests_24h": 1000000,
                "blocked_requests_24h": 5000,
                "detections_by_type": {
                    "sql_injection": 100,
                    "xss": 50,
                    "rate_limit": 4000,
                },
                "top_blocked_ips": [
                    {"ip": "45.33.32.156", "count": 500},
                ],
            }
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/metrics",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "total_requests_24h" in data
            assert "blocked_requests_24h" in data

    def test_get_waf_metrics_with_time_range(self, test_client, waf_viewer_headers):
        """Test getting WAF metrics with time range."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_metrics.return_value = {}
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/waf/metrics?start_time=2025-01-01T00:00:00Z&end_time=2025-01-31T23:59:59Z",
                headers=waf_viewer_headers,
            )

            assert response.status_code == 200


class TestRuleTestEndpoint:
    """Test rule testing endpoint."""

    def test_test_rule(self, test_client, waf_admin_headers):
        """Test testing a rule against sample traffic."""
        with patch(
            "greenlang.infrastructure.waf_management.api.waf_routes.get_waf_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.test_rule.return_value = {
                "matches": 5,
                "false_positives": 0,
                "sample_matches": [],
            }
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/waf/rules/test",
                headers=waf_admin_headers,
                json={
                    "rule_type": "sql_injection",
                    "parameters": {"sensitivity": "high"},
                    "test_samples": [
                        {"path": "/api/v1/users", "body": "admin' OR 1=1--"},
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "matches" in data
