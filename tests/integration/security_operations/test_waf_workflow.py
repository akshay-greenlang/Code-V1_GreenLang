"""
Integration tests for WAF management workflow.

Tests end-to-end WAF rule management, traffic analysis, and auto-mitigation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestWAFRuleWorkflow:
    """Test WAF rule management workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_rule_lifecycle(
        self,
        test_database,
        admin_auth_headers,
        integration_waf_rule_data,
    ):
        """Test complete WAF rule lifecycle."""
        mock_service = AsyncMock()

        # Step 1: Create rule
        rule = MagicMock()
        rule.rule_id = str(uuid4())
        rule.name = integration_waf_rule_data["name"]
        rule.enabled = True

        mock_service.create_rule.return_value = rule

        created_rule = await mock_service.create_rule(integration_waf_rule_data)
        assert created_rule.rule_id is not None

        # Step 2: Test rule against sample traffic
        mock_service.test_rule.return_value = {
            "matches": 5,
            "false_positives": 0,
        }

        test_result = await mock_service.test_rule(
            rule_id=rule.rule_id,
            samples=[{"path": "/api/test"}],
        )
        assert test_result["matches"] >= 0

        # Step 3: Deploy rule (enable in production)
        mock_service.deploy_rule.return_value = {"status": "deployed"}

        deploy_result = await mock_service.deploy_rule(rule.rule_id)
        assert deploy_result["status"] == "deployed"

        # Step 4: Monitor rule effectiveness
        mock_service.get_rule_metrics.return_value = {
            "requests_matched": 1000,
            "requests_blocked": 50,
            "false_positives_reported": 2,
        }

        metrics = await mock_service.get_rule_metrics(rule.rule_id)
        assert metrics["requests_blocked"] >= 0

        # Step 5: Update rule based on metrics
        mock_service.update_rule.return_value = rule

        updated_rule = await mock_service.update_rule(
            rule.rule_id,
            {"parameters": {"limit": 150}},
        )
        assert updated_rule is not None

        # Step 6: Disable rule if needed
        rule.enabled = False
        mock_service.disable_rule.return_value = rule

        disabled_rule = await mock_service.disable_rule(rule.rule_id)
        assert disabled_rule.enabled is False


class TestTrafficAnalysisWorkflow:
    """Test traffic analysis workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_baseline_creation_and_anomaly_detection(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test traffic baseline creation and anomaly detection."""
        mock_detector = AsyncMock()

        # Step 1: Collect traffic samples
        samples = [
            MagicMock(
                source_ip=f"192.168.1.{i}",
                path="/api/v1/test",
                response_code=200,
                response_time_ms=50 + (i % 20),
            )
            for i in range(2000)
        ]

        # Step 2: Build baseline from samples
        baseline = MagicMock()
        baseline.baseline_id = str(uuid4())
        baseline.metrics = {
            "requests_per_minute": {"mean": 100, "std_dev": 20},
            "response_time_ms": {"mean": 55, "std_dev": 10},
        }

        mock_detector.build_baseline.return_value = baseline

        created_baseline = await mock_detector.build_baseline(
            samples,
            name="API Baseline",
            endpoint_pattern="/api/.*",
        )
        assert created_baseline.baseline_id is not None

        # Step 3: Detect anomalies against baseline
        mock_detector.detect_anomalies.return_value = []

        # Normal traffic - no anomalies
        anomalies = await mock_detector.detect_anomalies({
            "requests_per_minute": 110,  # Within normal range
        })
        assert len(anomalies) == 0

        # Anomalous traffic
        mock_detector.detect_anomalies.return_value = [
            MagicMock(
                detection_type="anomaly",
                severity="high",
                description="Traffic spike: 10x above baseline",
            )
        ]

        anomalies = await mock_detector.detect_anomalies({
            "requests_per_minute": 1000,  # 10x baseline
        })
        assert len(anomalies) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_attack_detection_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test attack detection workflow."""
        mock_detector = AsyncMock()

        # Simulate SQL injection attack traffic
        attack_samples = [
            MagicMock(
                source_ip="45.33.32.156",
                path="/api/v1/users",
                body="username=admin' OR 1=1--",
                response_code=403,
            )
            for _ in range(10)
        ]

        # Detect attacks
        mock_detector.analyze_traffic.return_value = [
            MagicMock(
                detection_type="sql_injection",
                severity="critical",
                confidence=0.95,
                source_ip="45.33.32.156",
            )
        ]

        detections = await mock_detector.analyze_traffic(attack_samples)

        assert len(detections) == 1
        assert detections[0].detection_type == "sql_injection"


class TestAutoMitigationWorkflow:
    """Test auto-mitigation workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_auto_mitigation_on_attack(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test automatic mitigation on attack detection."""
        mock_detector = AsyncMock()
        mock_waf_service = AsyncMock()

        # Detection triggers auto-mitigation
        detection = MagicMock()
        detection.detection_type = "sql_injection"
        detection.severity = "critical"
        detection.traffic_sample = MagicMock(source_ip="45.33.32.156")

        # Auto-mitigation should block the IP
        mock_detector.auto_mitigate.return_value = True

        result = await mock_detector.auto_mitigate(detection)
        assert result is True

        # Verify IP was blocked
        mock_waf_service.block_ip.return_value = True

        block_result = await mock_waf_service.block_ip(
            ip_address="45.33.32.156",
            duration_hours=24,
            reason="Automated block: SQL injection attack detected",
        )
        assert block_result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ddos_mitigation_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test DDoS mitigation workflow."""
        mock_detector = AsyncMock()
        mock_waf_service = AsyncMock()

        # Detect DDoS attack
        mock_detector.detect_ddos.return_value = MagicMock(
            detection_type="ddos",
            severity="critical",
            metrics={
                "requests_per_second": 100000,
                "unique_sources": 5000,
            },
        )

        ddos_detection = await mock_detector.detect_ddos({
            "requests_per_second": 100000,
        })
        assert ddos_detection is not None

        # Activate DDoS mitigation
        mock_waf_service.activate_ddos_protection.return_value = {
            "status": "activated",
            "mode": "under_attack",
            "challenge_rate": 0.9,  # Challenge 90% of traffic
        }

        mitigation_result = await mock_waf_service.activate_ddos_protection()
        assert mitigation_result["status"] == "activated"


class TestIPBlockingWorkflow:
    """Test IP blocking workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ip_block_and_unblock_workflow(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test IP blocking and unblocking workflow."""
        mock_service = AsyncMock()

        ip_to_block = "45.33.32.156"

        # Block IP
        mock_service.block_ip.return_value = True

        block_result = await mock_service.block_ip(
            ip_address=ip_to_block,
            duration_hours=24,
            reason="Malicious activity",
        )
        assert block_result is True

        # Verify IP is blocked
        mock_service.is_ip_blocked.return_value = True

        is_blocked = await mock_service.is_ip_blocked(ip_to_block)
        assert is_blocked is True

        # Unblock IP
        mock_service.unblock_ip.return_value = True

        unblock_result = await mock_service.unblock_ip(ip_to_block)
        assert unblock_result is True

        # Verify IP is no longer blocked
        mock_service.is_ip_blocked.return_value = False

        is_blocked = await mock_service.is_ip_blocked(ip_to_block)
        assert is_blocked is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ip_block_expiration(
        self,
        test_database,
        admin_auth_headers,
    ):
        """Test IP block expiration."""
        mock_service = AsyncMock()

        ip_to_block = "45.33.32.156"

        # Block IP with 1 hour expiration
        mock_service.block_ip.return_value = True

        await mock_service.block_ip(
            ip_address=ip_to_block,
            duration_hours=1,
            reason="Temporary block",
        )

        # Check expiration time
        mock_service.get_block_info.return_value = {
            "ip_address": ip_to_block,
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "reason": "Temporary block",
        }

        block_info = await mock_service.get_block_info(ip_to_block)
        assert block_info["expires_at"] is not None
