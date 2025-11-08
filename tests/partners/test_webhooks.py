"""
Tests for Webhook System

Comprehensive tests for webhook delivery, retry logic,
signature verification, and security.
"""

import pytest
import json
import hmac
import hashlib
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from greenlang.partners.webhooks import (
    WebhookModel,
    WebhookDeliveryModel,
    WebhookEventType,
    WebhookStatus,
    DeliveryStatus,
    WebhookEvent,
    WebhookManager,
    create_webhook_app,
)
from greenlang.partners.webhook_security import (
    WebhookSignature,
    WebhookRateLimiter,
    IPWhitelist,
    WebhookValidator,
    SignatureVerificationError,
    ReplayAttackError,
    RateLimitExceededError,
)


# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_webhooks.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    from greenlang.partners.webhooks import Base
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_database):
    """Create database session for tests"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def webhook_manager(db_session):
    """Create webhook manager"""
    return WebhookManager(db_session)


@pytest.fixture
def sample_webhook(db_session):
    """Create sample webhook"""
    webhook = WebhookModel(
        id="wh_test123",
        partner_id="partner_123",
        url="https://example.com/webhook",
        secret="test_secret_key",
        status=WebhookStatus.ACTIVE,
        event_types=["workflow.completed", "agent.result"],
        max_retries=3,
        timeout_seconds=10
    )
    db_session.add(webhook)
    db_session.commit()
    db_session.refresh(webhook)
    return webhook


class TestWebhookSignature:
    """Tests for webhook signature generation and verification"""

    def test_generate_signature(self):
        """Test signature generation"""
        payload = b'{"event": "test"}'
        secret = "my_secret_key"
        timestamp = 1699459200

        signature = WebhookSignature.generate_signature(payload, secret, timestamp)

        assert signature.startswith("sha256=")
        assert len(signature) > 10

    def test_verify_signature_success(self):
        """Test successful signature verification"""
        payload = b'{"event": "test"}'
        secret = "my_secret_key"
        timestamp = int(time.time())

        signature = WebhookSignature.generate_signature(payload, secret, timestamp)

        # Should not raise exception
        result = WebhookSignature.verify_signature(
            payload,
            signature,
            secret,
            timestamp
        )
        assert result is True

    def test_verify_signature_invalid(self):
        """Test signature verification with invalid signature"""
        payload = b'{"event": "test"}'
        secret = "my_secret_key"
        timestamp = int(time.time())

        with pytest.raises(SignatureVerificationError):
            WebhookSignature.verify_signature(
                payload,
                "sha256=invalid_signature",
                secret,
                timestamp
            )

    def test_verify_signature_replay_attack(self):
        """Test signature verification detects replay attacks"""
        payload = b'{"event": "test"}'
        secret = "my_secret_key"
        old_timestamp = int(time.time()) - 400  # 400 seconds ago

        signature = WebhookSignature.generate_signature(payload, secret, old_timestamp)

        with pytest.raises(ReplayAttackError):
            WebhookSignature.verify_signature(
                payload,
                signature,
                secret,
                old_timestamp,
                tolerance_seconds=300  # 5 minutes
            )

    def test_verify_signature_wrong_secret(self):
        """Test signature verification with wrong secret"""
        payload = b'{"event": "test"}'
        secret = "my_secret_key"
        wrong_secret = "wrong_secret"
        timestamp = int(time.time())

        signature = WebhookSignature.generate_signature(payload, secret, timestamp)

        with pytest.raises(SignatureVerificationError):
            WebhookSignature.verify_signature(
                payload,
                signature,
                wrong_secret,
                timestamp
            )


class TestWebhookManager:
    """Tests for webhook manager"""

    def test_generate_signature(self, webhook_manager):
        """Test webhook signature generation"""
        payload = b'{"test": "data"}'
        secret = "test_secret"

        signature = webhook_manager.generate_signature(payload, secret)

        assert signature.startswith("sha256=")

    def test_verify_signature(self, webhook_manager):
        """Test webhook signature verification"""
        payload = b'{"test": "data"}'
        secret = "test_secret"

        signature = webhook_manager.generate_signature(payload, secret)
        result = webhook_manager.verify_signature(payload, signature, secret)

        assert result is True

    @pytest.mark.asyncio
    async def test_deliver_webhook_success(self, webhook_manager, sample_webhook):
        """Test successful webhook delivery"""
        event = WebhookEvent(
            event_type=WebhookEventType.WORKFLOW_COMPLETED,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={"workflow_id": "wf_123", "status": "success"}
        )

        # Mock HTTP request
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="OK")
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await webhook_manager.deliver_webhook(sample_webhook, event)

            assert result.success is True
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_deliver_webhook_failure(self, webhook_manager, sample_webhook):
        """Test webhook delivery failure"""
        event = WebhookEvent(
            event_type=WebhookEventType.WORKFLOW_COMPLETED,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={"test": "data"}
        )

        # Mock HTTP request with error
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Server Error")
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await webhook_manager.deliver_webhook(sample_webhook, event)

            assert result.success is False
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_deliver_webhook_timeout(self, webhook_manager, sample_webhook):
        """Test webhook delivery timeout"""
        event = WebhookEvent(
            event_type=WebhookEventType.WORKFLOW_COMPLETED,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={"test": "data"}
        )

        # Mock timeout
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()

            result = await webhook_manager.deliver_webhook(sample_webhook, event)

            assert result.success is False
            assert "Timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_send_event_creates_delivery(self, webhook_manager, sample_webhook, db_session):
        """Test that sending event creates delivery record"""
        event = WebhookEvent(
            event_type=WebhookEventType.WORKFLOW_COMPLETED,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={"test": "data"}
        )

        # Mock delivery
        with patch.object(webhook_manager, 'process_delivery', new=AsyncMock()):
            await webhook_manager.send_event("partner_123", event)

        # Check delivery record created
        deliveries = db_session.query(WebhookDeliveryModel).filter(
            WebhookDeliveryModel.webhook_id == sample_webhook.id
        ).all()

        assert len(deliveries) > 0


class TestWebhookRetry:
    """Tests for webhook retry logic"""

    @pytest.mark.asyncio
    async def test_retry_logic(self, webhook_manager, sample_webhook, db_session):
        """Test webhook retry logic"""
        # Create failed delivery
        delivery = WebhookDeliveryModel(
            id="del_retry_test",
            webhook_id=sample_webhook.id,
            event_type="workflow.completed",
            event_id="evt_123",
            url=sample_webhook.url,
            payload={"event": "test"},
            headers={},
            status=DeliveryStatus.FAILED,
            attempt_count=1,
            max_attempts=3
        )
        db_session.add(delivery)
        db_session.commit()

        # Mock successful retry
        with patch.object(webhook_manager, 'deliver_webhook', new=AsyncMock()) as mock_deliver:
            mock_deliver.return_value.success = True
            mock_deliver.return_value.status_code = 200

            await webhook_manager.process_delivery(delivery.id)

            # Should have been retried
            db_session.refresh(delivery)
            assert delivery.attempt_count == 2


class TestWebhookRateLimiting:
    """Tests for webhook rate limiting"""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter with mock Redis"""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.incr.return_value = 1
        return WebhookRateLimiter(mock_redis)

    def test_rate_limit_first_request(self, rate_limiter):
        """Test first request within window"""
        result = rate_limiter.check_rate_limit("partner_123", limit=100)
        assert result is True

    def test_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit exceeded"""
        # Mock Redis to return limit
        rate_limiter.redis.get.return_value = "100"

        with pytest.raises(RateLimitExceededError):
            rate_limiter.check_rate_limit("partner_123", limit=100)

    def test_get_remaining_quota(self, rate_limiter):
        """Test getting remaining quota"""
        rate_limiter.redis.get.return_value = "50"

        remaining = rate_limiter.get_remaining_quota("partner_123", limit=100)
        assert remaining == 50


class TestIPWhitelist:
    """Tests for IP whitelisting"""

    @pytest.fixture
    def ip_whitelist(self):
        """Create IP whitelist"""
        return IPWhitelist()

    def test_add_ip_range(self, ip_whitelist):
        """Test adding IP range"""
        ip_whitelist.add_ip_range("partner_123", "192.168.1.0/24")

        whitelist = ip_whitelist.get_whitelist("partner_123")
        assert "192.168.1.0/24" in whitelist

    def test_is_allowed_with_whitelist(self, ip_whitelist):
        """Test IP allowed with whitelist"""
        ip_whitelist.add_ip_range("partner_123", "192.168.1.0/24")

        assert ip_whitelist.is_allowed("partner_123", "192.168.1.100") is True
        assert ip_whitelist.is_allowed("partner_123", "10.0.0.1") is False

    def test_is_allowed_without_whitelist(self, ip_whitelist):
        """Test IP allowed without whitelist (all allowed)"""
        assert ip_whitelist.is_allowed("partner_123", "1.2.3.4") is True

    def test_remove_ip_range(self, ip_whitelist):
        """Test removing IP range"""
        ip_whitelist.add_ip_range("partner_123", "192.168.1.0/24")
        ip_whitelist.remove_ip_range("partner_123", "192.168.1.0/24")

        whitelist = ip_whitelist.get_whitelist("partner_123")
        assert len(whitelist) == 0


class TestWebhookValidator:
    """Tests for webhook validator"""

    def test_validate_payload_size_success(self):
        """Test payload size validation success"""
        payload = b"small payload"
        result = WebhookValidator.validate_payload_size(payload, max_size=1000)
        assert result is True

    def test_validate_payload_size_exceeded(self):
        """Test payload size validation exceeded"""
        payload = b"x" * 2000
        with pytest.raises(ValueError):
            WebhookValidator.validate_payload_size(payload, max_size=1000)


class TestWebhookDeliveryLog:
    """Tests for webhook delivery logging"""

    def test_delivery_log_creation(self, db_session, sample_webhook):
        """Test creating delivery log"""
        delivery = WebhookDeliveryModel(
            id="del_log_test",
            webhook_id=sample_webhook.id,
            event_type="workflow.completed",
            event_id="evt_log_123",
            url=sample_webhook.url,
            payload={"test": "data"},
            headers={"Content-Type": "application/json"},
            status=DeliveryStatus.PENDING,
            attempt_count=0,
            max_attempts=3
        )

        db_session.add(delivery)
        db_session.commit()

        # Verify created
        saved_delivery = db_session.query(WebhookDeliveryModel).filter(
            WebhookDeliveryModel.id == "del_log_test"
        ).first()

        assert saved_delivery is not None
        assert saved_delivery.event_type == "workflow.completed"
        assert saved_delivery.status == DeliveryStatus.PENDING

    def test_delivery_log_update_success(self, db_session, sample_webhook):
        """Test updating delivery log on success"""
        delivery = WebhookDeliveryModel(
            id="del_success_test",
            webhook_id=sample_webhook.id,
            event_type="workflow.completed",
            event_id="evt_success",
            url=sample_webhook.url,
            payload={"test": "data"},
            headers={},
            status=DeliveryStatus.PENDING
        )
        db_session.add(delivery)
        db_session.commit()

        # Update to success
        delivery.status = DeliveryStatus.SUCCESS
        delivery.status_code = 200
        delivery.response_time_ms = 150
        delivery.completed_at = datetime.utcnow()
        db_session.commit()

        # Verify updated
        db_session.refresh(delivery)
        assert delivery.status == DeliveryStatus.SUCCESS
        assert delivery.status_code == 200


class TestWebhookStatistics:
    """Tests for webhook statistics"""

    def test_webhook_statistics_tracking(self, db_session, sample_webhook):
        """Test webhook statistics tracking"""
        # Simulate deliveries
        sample_webhook.total_deliveries = 100
        sample_webhook.successful_deliveries = 95
        sample_webhook.failed_deliveries = 5
        db_session.commit()

        # Verify statistics
        assert sample_webhook.total_deliveries == 100
        assert sample_webhook.successful_deliveries == 95
        assert sample_webhook.failed_deliveries == 5

        # Calculate success rate
        success_rate = (sample_webhook.successful_deliveries / sample_webhook.total_deliveries) * 100
        assert success_rate == 95.0


class TestWebhookEventTypes:
    """Tests for different webhook event types"""

    def test_workflow_completed_event(self):
        """Test workflow completed event"""
        event = WebhookEvent(
            event_type=WebhookEventType.WORKFLOW_COMPLETED,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={
                "workflow_id": "wf_123",
                "status": "success",
                "duration_ms": 1500
            }
        )

        assert event.event_type == WebhookEventType.WORKFLOW_COMPLETED
        assert event.data["workflow_id"] == "wf_123"

    def test_agent_result_event(self):
        """Test agent result event"""
        event = WebhookEvent(
            event_type=WebhookEventType.AGENT_RESULT,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={
                "agent_id": "carbon_analyzer",
                "result": {"emissions": 100},
                "confidence": 0.95
            }
        )

        assert event.event_type == WebhookEventType.AGENT_RESULT
        assert event.data["agent_id"] == "carbon_analyzer"

    def test_usage_limit_reached_event(self):
        """Test usage limit reached event"""
        event = WebhookEvent(
            event_type=WebhookEventType.USAGE_LIMIT_REACHED,
            timestamp=datetime.utcnow(),
            partner_id="partner_123",
            data={
                "current_usage": 1000,
                "quota_limit": 1000,
                "period": "hour"
            }
        )

        assert event.event_type == WebhookEventType.USAGE_LIMIT_REACHED
        assert event.data["current_usage"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
