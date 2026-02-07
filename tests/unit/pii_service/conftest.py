# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for PII Service unit tests.

Provides comprehensive fixtures for testing all PII Service components:
- Mock encryption service (SEC-003 integration)
- Mock audit service
- Configuration fixtures for vault, enforcement, allowlist
- Sample detection data generators
- Test tenant and user context
- Mock database and storage backends

Author: GreenLang Test Engineering Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest


# ============================================================================
# Encryption Service Fixtures
# ============================================================================


@pytest.fixture
def mock_encryption_service():
    """Mock SEC-003 EncryptionService for vault testing.

    Provides encrypt/decrypt operations that simulate AES-256-GCM behavior
    without actual cryptographic operations.
    """
    service = AsyncMock()

    # Track encrypted values for verification
    _encrypted_store: Dict[bytes, bytes] = {}

    async def mock_encrypt(data: bytes, key_id: str = None) -> bytes:
        """Simulate encryption - prefix with marker for testing."""
        encrypted = b"ENC:" + data
        _encrypted_store[encrypted] = data
        return encrypted

    async def mock_decrypt(encrypted_data: bytes, key_id: str = None) -> bytes:
        """Simulate decryption - strip prefix."""
        if encrypted_data.startswith(b"ENC:"):
            return encrypted_data[4:]
        raise ValueError("Invalid encrypted data format")

    service.encrypt = AsyncMock(side_effect=mock_encrypt)
    service.decrypt = AsyncMock(side_effect=mock_decrypt)
    service.rotate_key = AsyncMock(return_value=True)
    service.get_key_metadata = AsyncMock(return_value={
        "key_id": "test-key",
        "algorithm": "AES-256-GCM",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    service._encrypted_store = _encrypted_store

    return service


@pytest.fixture
def mock_encryption_service_failing():
    """Mock EncryptionService that fails operations for error testing."""
    service = AsyncMock()
    service.encrypt = AsyncMock(side_effect=Exception("Encryption failed"))
    service.decrypt = AsyncMock(side_effect=Exception("Decryption failed"))
    return service


# ============================================================================
# Audit Service Fixtures
# ============================================================================


@pytest.fixture
def mock_audit_service():
    """Mock audit service for tracking operations."""
    service = AsyncMock()
    service._audit_log: List[Dict[str, Any]] = []

    async def mock_log(event_type: str, **kwargs):
        service._audit_log.append({
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        })

    service.log_event = AsyncMock(side_effect=mock_log)
    service.log_access = AsyncMock(side_effect=lambda **kw: mock_log("access", **kw))
    service.log_denied = AsyncMock(side_effect=lambda **kw: mock_log("access_denied", **kw))
    service.get_audit_log = lambda: service._audit_log

    return service


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def vault_config():
    """Create VaultConfig for testing."""
    try:
        from greenlang.infrastructure.pii_service.config import (
            VaultConfig,
            PersistenceBackend,
        )
        return VaultConfig(
            token_ttl_days=90,
            max_tokens_per_tenant=1_000_000,
            persistence_backend=PersistenceBackend.MEMORY,
            encryption_key_id="test-key",
            enable_persistence=True,
            cache_ttl_seconds=300,
            cache_max_size=10000,
            require_user_auth=True,
        )
    except ImportError:
        # Return a simple stub if module not available
        class VaultConfigStub:
            token_ttl_days = 90
            max_tokens_per_tenant = 1_000_000
            persistence_backend = "memory"
            encryption_key_id = "test-key"
            enable_persistence = True
            cache_ttl_seconds = 300
            cache_max_size = 10000
            require_user_auth = True
        return VaultConfigStub()


@pytest.fixture
def enforcement_config():
    """Create EnforcementConfig for testing."""
    try:
        from greenlang.infrastructure.pii_service.config import (
            EnforcementConfig,
            EnforcementMode,
        )
        return EnforcementConfig(
            mode=EnforcementMode.ENFORCE,
            scan_requests=True,
            scan_responses=False,
            scan_logs=True,
            min_confidence=0.8,
            block_high_sensitivity=True,
            quarantine_enabled=True,
            quarantine_ttl_hours=72,
            notification_enabled=True,
            exclude_paths=["/health", "/metrics"],
        )
    except ImportError:
        class EnforcementConfigStub:
            mode = "enforce"
            scan_requests = True
            scan_responses = False
            scan_logs = True
            min_confidence = 0.8
            block_high_sensitivity = True
            quarantine_enabled = True
            quarantine_ttl_hours = 72
            notification_enabled = True
            exclude_paths = ["/health", "/metrics"]
        return EnforcementConfigStub()


@pytest.fixture
def allowlist_config():
    """Create AllowlistConfig for testing."""
    try:
        from greenlang.infrastructure.pii_service.config import AllowlistConfig
        return AllowlistConfig(
            enable_defaults=True,
            cache_ttl_seconds=60,
            max_entries_per_tenant=10000,
            require_reason=True,
            expiration_days=0,
        )
    except ImportError:
        class AllowlistConfigStub:
            enable_defaults = True
            cache_ttl_seconds = 60
            max_entries_per_tenant = 10000
            require_reason = True
            expiration_days = 0
        return AllowlistConfigStub()


@pytest.fixture
def streaming_config():
    """Create StreamingConfig for testing."""
    try:
        from greenlang.infrastructure.pii_service.config import (
            StreamingConfig,
            StreamingPlatform,
        )
        return StreamingConfig(
            enabled=True,
            platform=StreamingPlatform.KAFKA,
            bootstrap_servers="localhost:9092",
            input_topics=["raw-events"],
            output_topic="clean-events",
            dlq_topic="pii-blocked",
            consumer_group="pii-scanner-test",
            batch_size=100,
            batch_timeout_ms=1000,
        )
    except ImportError:
        class StreamingConfigStub:
            enabled = True
            platform = "kafka"
            bootstrap_servers = "localhost:9092"
            input_topics = ["raw-events"]
            output_topic = "clean-events"
            dlq_topic = "pii-blocked"
            consumer_group = "pii-scanner-test"
            batch_size = 100
            batch_timeout_ms = 1000
        return StreamingConfigStub()


@pytest.fixture
def remediation_config():
    """Create RemediationConfig for testing."""
    try:
        from greenlang.infrastructure.pii_service.config import (
            RemediationConfig,
            RemediationAction,
        )
        return RemediationConfig(
            enabled=True,
            default_action=RemediationAction.NOTIFY_ONLY,
            delay_hours=72,
            requires_approval=True,
            notify_on_action=True,
            batch_size=100,
            dry_run=False,
            generate_certificates=True,
        )
    except ImportError:
        class RemediationConfigStub:
            enabled = True
            default_action = "notify_only"
            delay_hours = 72
            requires_approval = True
            notify_on_action = True
            batch_size = 100
            dry_run = False
            generate_certificates = True
        return RemediationConfigStub()


@pytest.fixture
def pii_service_config(vault_config, enforcement_config, allowlist_config):
    """Create full PIIServiceConfig for testing."""
    try:
        from greenlang.infrastructure.pii_service.config import PIIServiceConfig
        return PIIServiceConfig(
            service_name="pii_service_test",
            enable_metrics=True,
            enable_audit=True,
            vault=vault_config,
            enforcement=enforcement_config,
            allowlist=allowlist_config,
        )
    except ImportError:
        class PIIServiceConfigStub:
            service_name = "pii_service_test"
            enable_metrics = True
            enable_audit = True
        config = PIIServiceConfigStub()
        config.vault = vault_config
        config.enforcement = enforcement_config
        config.allowlist = allowlist_config
        return config


# ============================================================================
# PII Type and Detection Fixtures
# ============================================================================


@pytest.fixture
def pii_type_enum():
    """Get PIIType enum for tests."""
    try:
        from greenlang.infrastructure.pii_service.models import PIIType
        return PIIType
    except ImportError:
        from enum import Enum
        class PIIType(str, Enum):
            SSN = "ssn"
            EMAIL = "email"
            PHONE = "phone"
            CREDIT_CARD = "credit_card"
            PERSON_NAME = "person_name"
            ADDRESS = "address"
            IP_ADDRESS = "ip_address"
            API_KEY = "api_key"
            PASSWORD = "password"
        return PIIType


@pytest.fixture
def sample_detections(pii_type_enum):
    """Create sample PII detection results."""
    try:
        from greenlang.infrastructure.pii_service.models import PIIDetection
        return [
            PIIDetection(
                pii_type=pii_type_enum.SSN,
                value_hash=hashlib.sha256(b"123-45-6789").hexdigest(),
                confidence=0.95,
                start=10,
                end=21,
                context="SSN: ***-**-****",
            ),
            PIIDetection(
                pii_type=pii_type_enum.EMAIL,
                value_hash=hashlib.sha256(b"john@example.com").hexdigest(),
                confidence=0.90,
                start=30,
                end=50,
                context="email: ***@***.com",
            ),
            PIIDetection(
                pii_type=pii_type_enum.PHONE,
                value_hash=hashlib.sha256(b"555-123-4567").hexdigest(),
                confidence=0.85,
                start=60,
                end=72,
                context="phone: ***-***-****",
            ),
        ]
    except ImportError:
        # Return simple dict representations
        return [
            {
                "pii_type": "ssn",
                "value_hash": hashlib.sha256(b"123-45-6789").hexdigest(),
                "confidence": 0.95,
                "start": 10,
                "end": 21,
                "context": "SSN: ***-**-****",
            },
            {
                "pii_type": "email",
                "value_hash": hashlib.sha256(b"john@example.com").hexdigest(),
                "confidence": 0.90,
                "start": 30,
                "end": 50,
                "context": "email: ***@***.com",
            },
            {
                "pii_type": "phone",
                "value_hash": hashlib.sha256(b"555-123-4567").hexdigest(),
                "confidence": 0.85,
                "start": 60,
                "end": 72,
                "context": "phone: ***-***-****",
            },
        ]


@pytest.fixture
def sample_detection_ssn(pii_type_enum):
    """Create a single SSN detection."""
    try:
        from greenlang.infrastructure.pii_service.models import PIIDetection
        return PIIDetection(
            pii_type=pii_type_enum.SSN,
            value_hash=hashlib.sha256(b"123-45-6789").hexdigest(),
            confidence=0.95,
            start=10,
            end=21,
            context="SSN: 123-**-****",
        )
    except ImportError:
        return {
            "pii_type": "ssn",
            "value_hash": hashlib.sha256(b"123-45-6789").hexdigest(),
            "confidence": 0.95,
            "start": 10,
            "end": 21,
        }


@pytest.fixture
def sample_detection_email(pii_type_enum):
    """Create a single email detection."""
    try:
        from greenlang.infrastructure.pii_service.models import PIIDetection
        return PIIDetection(
            pii_type=pii_type_enum.EMAIL,
            value_hash=hashlib.sha256(b"john@example.com").hexdigest(),
            confidence=0.90,
            start=30,
            end=46,
            context="email: j***@example.com",
        )
    except ImportError:
        return {
            "pii_type": "email",
            "value_hash": hashlib.sha256(b"john@example.com").hexdigest(),
            "confidence": 0.90,
            "start": 30,
            "end": 46,
        }


# ============================================================================
# Content and Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_content():
    """Sample content containing multiple PII types."""
    return "My SSN is 123-45-6789 and email is john@example.com. Call me at 555-123-4567."


@pytest.fixture
def sample_content_no_pii():
    """Sample content with no PII."""
    return "This is a clean message with no sensitive information."


@pytest.fixture
def sample_content_high_sensitivity():
    """Sample content with high-sensitivity PII."""
    return "Credit card: 4111-1111-1111-1111, SSN: 123-45-6789, Password: secret123!"


@pytest.fixture
def sample_content_test_data():
    """Sample content with test/allowlisted data."""
    return "Email: test@example.com, Phone: 555-555-5555, Card: 4242424242424242"


@pytest.fixture
def sample_json_content():
    """Sample JSON content with PII."""
    return json.dumps({
        "user": {
            "name": "John Doe",
            "email": "john.doe@company.com",
            "ssn": "123-45-6789",
            "phone": "+1-555-123-4567",
        },
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "zip": "12345",
        },
    })


# ============================================================================
# Tenant and User Context Fixtures
# ============================================================================


@pytest.fixture
def test_tenant_id():
    """Standard test tenant ID."""
    return "tenant-test-123"


@pytest.fixture
def test_tenant_id_alt():
    """Alternative test tenant ID for isolation tests."""
    return "tenant-test-456"


@pytest.fixture
def test_user_id():
    """Standard test user ID."""
    return str(uuid4())


@pytest.fixture
def test_user_id_admin():
    """Admin user ID for privileged operations."""
    return str(uuid4())


@pytest.fixture
def test_request_id():
    """Standard test request correlation ID."""
    return str(uuid4())


@pytest.fixture
def enforcement_context(test_tenant_id, test_user_id, test_request_id):
    """Create EnforcementContext for testing."""
    try:
        from greenlang.infrastructure.pii_service.models import EnforcementContext
        return EnforcementContext(
            context_type="api_request",
            path="/api/v1/data",
            method="POST",
            tenant_id=test_tenant_id,
            user_id=test_user_id,
            request_id=test_request_id,
        )
    except ImportError:
        class ContextStub:
            context_type = "api_request"
            path = "/api/v1/data"
            method = "POST"
            timestamp = datetime.now(timezone.utc)
        ctx = ContextStub()
        ctx.tenant_id = test_tenant_id
        ctx.user_id = test_user_id
        ctx.request_id = test_request_id
        return ctx


# ============================================================================
# Database and Storage Fixtures
# ============================================================================


@pytest.fixture
def mock_db_pool():
    """Mock async database connection pool."""
    pool = AsyncMock()

    # In-memory storage for tokens
    _token_store: Dict[str, Dict[str, Any]] = {}

    async def mock_execute(query: str, *args):
        return MagicMock(rowcount=1)

    async def mock_fetchone(query: str, *args):
        if "token_vault" in query and args:
            token_id = args[0]
            return _token_store.get(token_id)
        return None

    async def mock_fetchall(query: str, *args):
        return list(_token_store.values())

    pool.execute = mock_execute
    pool.fetchone = mock_fetchone
    pool.fetchall = mock_fetchall
    pool._token_store = _token_store

    return pool


@pytest.fixture
def mock_redis_client():
    """Mock Redis client with cache behavior."""
    cache: Dict[str, str] = {}

    client = AsyncMock()

    async def mock_get(key: str):
        return cache.get(key)

    async def mock_set(key: str, value: str, ex: Optional[int] = None):
        cache[key] = value

    async def mock_delete(*keys: str):
        for key in keys:
            cache.pop(key, None)

    async def mock_keys(pattern: str):
        regex = pattern.replace("*", ".*")
        return [k for k in cache.keys() if re.match(regex, k)]

    async def mock_publish(channel: str, message: str):
        pass

    client.get = mock_get
    client.set = mock_set
    client.delete = mock_delete
    client.keys = mock_keys
    client.publish = mock_publish
    client.close = AsyncMock()
    client._test_cache = cache

    return client


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for storage testing."""
    _object_store: Dict[str, bytes] = {}

    client = AsyncMock()

    async def mock_put_object(Bucket: str, Key: str, Body: bytes, **kwargs):
        _object_store[f"{Bucket}/{Key}"] = Body
        return {"ETag": hashlib.md5(Body).hexdigest()}

    async def mock_get_object(Bucket: str, Key: str, **kwargs):
        key = f"{Bucket}/{Key}"
        if key in _object_store:
            return {"Body": AsyncMock(read=AsyncMock(return_value=_object_store[key]))}
        raise Exception("NoSuchKey")

    async def mock_delete_object(Bucket: str, Key: str, **kwargs):
        key = f"{Bucket}/{Key}"
        _object_store.pop(key, None)

    client.put_object = mock_put_object
    client.get_object = mock_get_object
    client.delete_object = mock_delete_object
    client._object_store = _object_store

    return client


# ============================================================================
# Kafka/Kinesis Fixtures
# ============================================================================


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for streaming tests."""
    producer = AsyncMock()
    producer._sent_messages: List[Dict[str, Any]] = []

    async def mock_send_and_wait(topic: str, value: bytes, headers: list = None):
        producer._sent_messages.append({
            "topic": topic,
            "value": value,
            "headers": headers or [],
        })

    producer.send_and_wait = mock_send_and_wait
    producer.start = AsyncMock()
    producer.stop = AsyncMock()

    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer for streaming tests."""
    consumer = AsyncMock()
    consumer._messages: List[Dict[str, Any]] = []
    consumer._position = 0

    async def mock_getone():
        if consumer._position < len(consumer._messages):
            msg = consumer._messages[consumer._position]
            consumer._position += 1
            return MagicMock(
                topic=msg.get("topic", "test-topic"),
                value=msg.get("value", b"{}"),
                headers=msg.get("headers", []),
            )
        return None

    consumer.getone = mock_getone
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.subscribe = AsyncMock()

    return consumer


@pytest.fixture
def mock_kinesis_client():
    """Mock Kinesis client for streaming tests."""
    client = AsyncMock()
    client._records: List[Dict[str, Any]] = []

    async def mock_put_record(StreamName: str, Data: bytes, PartitionKey: str):
        client._records.append({
            "stream": StreamName,
            "data": Data,
            "partition_key": PartitionKey,
        })
        return {"SequenceNumber": str(len(client._records)), "ShardId": "shard-1"}

    async def mock_get_records(ShardIterator: str, Limit: int = 100):
        return {
            "Records": [
                {"Data": r["data"], "SequenceNumber": str(i)}
                for i, r in enumerate(client._records[:Limit])
            ],
            "NextShardIterator": "next-iterator",
        }

    client.put_record = mock_put_record
    client.get_records = mock_get_records

    return client


# ============================================================================
# Notification Fixtures
# ============================================================================


@pytest.fixture
def mock_notification_service():
    """Mock notification service for alert testing."""
    service = AsyncMock()
    service._notifications: List[Dict[str, Any]] = []

    async def mock_send(channel: str, message: str, **kwargs):
        service._notifications.append({
            "channel": channel,
            "message": message,
            **kwargs,
        })

    service.send = mock_send
    service.send_alert = AsyncMock(side_effect=lambda **kw: mock_send("alert", **kw))
    service.send_email = AsyncMock(side_effect=lambda **kw: mock_send("email", **kw))

    return service


# ============================================================================
# Allowlist Fixtures
# ============================================================================


@pytest.fixture
def sample_allowlist_entry(pii_type_enum, test_user_id):
    """Create sample allowlist entry."""
    try:
        from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry
        return AllowlistEntry(
            pii_type=pii_type_enum.EMAIL,
            pattern=r".*@example\.com$",
            pattern_type="regex",
            reason="RFC 2606 reserved domain",
            created_by=UUID(test_user_id),
        )
    except ImportError:
        return {
            "pii_type": "email",
            "pattern": r".*@example\.com$",
            "pattern_type": "regex",
            "reason": "RFC 2606 reserved domain",
            "created_by": test_user_id,
        }


@pytest.fixture
def sample_allowlist_entries(pii_type_enum, test_user_id):
    """Create multiple sample allowlist entries."""
    try:
        from greenlang.infrastructure.pii_service.allowlist.patterns import (
            AllowlistEntry,
            PatternType,
        )
        return [
            AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@example\.com$",
                pattern_type=PatternType.REGEX,
                reason="RFC 2606 reserved domain",
                created_by=UUID(test_user_id),
            ),
            AllowlistEntry(
                pii_type=pii_type_enum.PHONE,
                pattern="555-555-5555",
                pattern_type=PatternType.EXACT,
                reason="Fictional phone number",
                created_by=UUID(test_user_id),
            ),
            AllowlistEntry(
                pii_type=pii_type_enum.CREDIT_CARD,
                pattern="4242424242424242",
                pattern_type=PatternType.EXACT,
                reason="Stripe test card",
                created_by=UUID(test_user_id),
            ),
        ]
    except ImportError:
        return [
            {"pii_type": "email", "pattern": r".*@example\.com$"},
            {"pii_type": "phone", "pattern": "555-555-5555"},
            {"pii_type": "credit_card", "pattern": "4242424242424242"},
        ]


# ============================================================================
# Remediation Fixtures
# ============================================================================


@pytest.fixture
def sample_remediation_item(test_tenant_id, pii_type_enum):
    """Create sample remediation item."""
    return {
        "id": str(uuid4()),
        "pii_type": pii_type_enum.SSN if hasattr(pii_type_enum, "SSN") else "ssn",
        "source_type": "database",
        "source_location": "users.ssn",
        "tenant_id": test_tenant_id,
        "status": "pending",
        "scheduled_at": (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat(),
        "action": "delete",
    }


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================


@pytest.fixture
def auth_headers(test_tenant_id, test_user_id):
    """Generate authentication headers for API tests."""
    def _generate(
        tenant_id: str = None,
        user_id: str = None,
        roles: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        roles = roles or ["pii:read", "pii:write"]
        return {
            "Authorization": "Bearer test-jwt-token",
            "X-Tenant-ID": tenant_id or test_tenant_id,
            "X-User-ID": user_id or test_user_id,
            "X-Roles": ",".join(roles),
        }
    return _generate


@pytest.fixture
def admin_auth_headers(auth_headers):
    """Generate admin authentication headers."""
    return auth_headers(
        roles=["pii:admin", "pii:read", "pii:write", "pii:tokenize", "pii:detokenize"],
    )


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture
def generate_pii_content():
    """Factory for generating content with specific PII types."""
    def _generate(
        include_ssn: bool = False,
        include_email: bool = False,
        include_phone: bool = False,
        include_credit_card: bool = False,
        include_name: bool = False,
    ) -> str:
        parts = ["Some text"]
        if include_ssn:
            parts.append("SSN: 123-45-6789")
        if include_email:
            parts.append("email: john@company.com")
        if include_phone:
            parts.append("phone: 555-123-4567")
        if include_credit_card:
            parts.append("card: 4111-1111-1111-1111")
        if include_name:
            parts.append("name: John Smith")
        return ". ".join(parts) + "."
    return _generate


@pytest.fixture
def generate_token_id():
    """Generate unique token IDs for testing."""
    def _generate(prefix: str = "tok") -> str:
        return f"{prefix}_{uuid4().hex[:16]}"
    return _generate


@pytest.fixture
def generate_batch_content():
    """Generate batch of content with varying PII types."""
    def _generate(count: int = 100) -> List[str]:
        templates = [
            "User {} has email {}@company.com",
            "SSN for record {} is 123-45-{}",
            "Phone number: 555-{}-{}",
            "Name: Test User {}, card ending in {}",
            "Address: {} Main St, City {}",
        ]
        return [
            templates[i % len(templates)].format(i, str(i).zfill(4))
            for i in range(count)
        ]
    return _generate


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers for PII service tests."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as a compliance/regulatory test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )


@pytest.fixture(autouse=True)
def reset_context():
    """Reset any context variables between tests."""
    yield
    # Cleanup after each test


@pytest.fixture
def freeze_time():
    """Fixture to freeze time for deterministic tests."""
    from unittest.mock import patch

    def _freeze(target_time: datetime):
        return patch(
            "datetime.datetime",
            wraps=datetime,
            now=lambda tz=None: target_time,
            utcnow=lambda: target_time.replace(tzinfo=None),
        )
    return _freeze
