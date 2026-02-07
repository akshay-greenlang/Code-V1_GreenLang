# -*- coding: utf-8 -*-
"""
Unit tests for PIIRemediationEngine - SEC-011 PII Service.

Tests the auto-remediation engine for PII cleanup:
- Remediation scheduling and execution
- Delete, anonymize, archive actions
- Grace period handling
- Approval workflows
- Deletion certificate generation
- Notification integration

Coverage target: 85%+ of remediation/*.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def remediation_engine(remediation_config, mock_db_pool, mock_audit_service, mock_notification_service):
    """Create PIIRemediationEngine instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.remediation.engine import PIIRemediationEngine
        return PIIRemediationEngine(
            config=remediation_config,
            db_pool=mock_db_pool,
            audit_service=mock_audit_service,
            notification_service=mock_notification_service,
        )
    except ImportError:
        pytest.skip("PIIRemediationEngine not yet implemented")


@pytest.fixture
def remediation_engine_dry_run(remediation_config, mock_db_pool, mock_audit_service, mock_notification_service):
    """Create PIIRemediationEngine in dry-run mode."""
    try:
        from greenlang.infrastructure.pii_service.remediation.engine import PIIRemediationEngine
        remediation_config.dry_run = True
        return PIIRemediationEngine(
            config=remediation_config,
            db_pool=mock_db_pool,
            audit_service=mock_audit_service,
            notification_service=mock_notification_service,
        )
    except ImportError:
        pytest.skip("PIIRemediationEngine not yet implemented")


@pytest.fixture
def sample_pii_item(test_tenant_id, pii_type_enum):
    """Create sample PII remediation item."""
    return {
        "id": str(uuid4()),
        "pii_type": pii_type_enum.SSN if hasattr(pii_type_enum, "SSN") else "ssn",
        "source_type": "database",
        "source_location": "users.ssn_encrypted",
        "tenant_id": test_tenant_id,
        "value_hash": "abc123def456",
        "detected_at": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
        "scheduled_for": (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat(),
        "status": "pending",
        "action": "delete",
    }


@pytest.fixture
def mock_source_handler():
    """Mock source-specific handler."""
    handler = AsyncMock()
    handler.delete = AsyncMock(return_value=True)
    handler.anonymize = AsyncMock(return_value=True)
    handler.archive = AsyncMock(return_value=True)
    handler.exists = AsyncMock(return_value=True)
    return handler


# ============================================================================
# TestRemediationScheduling
# ============================================================================


class TestRemediationScheduling:
    """Tests for remediation scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_remediation_creates_item(
        self, remediation_engine, pii_type_enum, test_tenant_id
    ):
        """schedule_remediation() creates a pending item."""
        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        assert item_id is not None

        # Item should be retrievable
        item = await remediation_engine.get_item(item_id)
        assert item is not None
        assert item.status == "pending"

    @pytest.mark.asyncio
    async def test_schedule_remediation_sets_delay(
        self, remediation_engine, remediation_config, pii_type_enum, test_tenant_id
    ):
        """schedule_remediation() respects configured delay."""
        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        item = await remediation_engine.get_item(item_id)

        # Scheduled time should be delay_hours in the future
        expected_time = datetime.now(timezone.utc) + timedelta(
            hours=remediation_config.delay_hours
        )
        actual_time = datetime.fromisoformat(str(item.scheduled_for))

        # Within 1 minute tolerance
        assert abs((actual_time - expected_time).total_seconds()) < 60

    @pytest.mark.asyncio
    async def test_schedule_remediation_with_custom_action(
        self, remediation_engine, pii_type_enum, test_tenant_id
    ):
        """schedule_remediation() accepts custom action."""
        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.EMAIL,
            source_type="database",
            source_location="users.email",
            tenant_id=test_tenant_id,
            value_hash="def456",
            action="anonymize",
        )

        item = await remediation_engine.get_item(item_id)
        assert item.action == "anonymize"


# ============================================================================
# TestProcessPending
# ============================================================================


class TestProcessPending:
    """Tests for processing pending remediations."""

    @pytest.mark.asyncio
    async def test_process_pending_respects_grace_period(
        self, remediation_engine, pii_type_enum, test_tenant_id
    ):
        """process_pending() only processes items past grace period."""
        # Schedule item for future
        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        # Process pending
        processed = await remediation_engine.process_pending()

        # Item should not be processed yet (still in grace period)
        item = await remediation_engine.get_item(item_id)
        assert item.status == "pending"

    @pytest.mark.asyncio
    async def test_process_pending_executes_due_items(
        self, remediation_engine, pii_type_enum, test_tenant_id
    ):
        """process_pending() executes items past their scheduled time."""
        # Schedule item for past (immediate)
        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        # Manually set scheduled time to past
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        # Process pending
        processed = await remediation_engine.process_pending()

        # Item should be processed
        assert processed >= 1

    @pytest.mark.asyncio
    async def test_process_pending_requires_approval(
        self, remediation_engine, remediation_config, pii_type_enum, test_tenant_id
    ):
        """process_pending() skips items requiring approval if not approved."""
        remediation_config.requires_approval = True

        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        # Set as due
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        # Process without approval
        processed = await remediation_engine.process_pending()

        # Item should not be processed (needs approval)
        item = await remediation_engine.get_item(item_id)
        assert item.status == "pending" or item.status == "awaiting_approval"


# ============================================================================
# TestRemediationActions
# ============================================================================


class TestRemediationActions:
    """Tests for remediation action execution."""

    @pytest.mark.asyncio
    async def test_delete_action_removes_from_source(
        self, remediation_engine, mock_source_handler, sample_pii_item
    ):
        """Delete action removes PII from source system."""
        # Register source handler
        remediation_engine._register_source_handler("database", mock_source_handler)

        # Create and execute item
        item_id = await remediation_engine._create_item(sample_pii_item)
        sample_pii_item["action"] = "delete"

        await remediation_engine._execute_action(item_id)

        mock_source_handler.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anonymize_action_masks_data(
        self, remediation_engine, mock_source_handler, sample_pii_item
    ):
        """Anonymize action masks PII in source system."""
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "anonymize"
        item_id = await remediation_engine._create_item(sample_pii_item)

        await remediation_engine._execute_action(item_id)

        mock_source_handler.anonymize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_archive_action_copies_then_deletes(
        self, remediation_engine, mock_source_handler, sample_pii_item
    ):
        """Archive action copies to archive then deletes from source."""
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "archive"
        item_id = await remediation_engine._create_item(sample_pii_item)

        await remediation_engine._execute_action(item_id)

        # Should archive then delete
        mock_source_handler.archive.assert_awaited_once()


# ============================================================================
# TestDeletionCertificates
# ============================================================================


class TestDeletionCertificates:
    """Tests for deletion certificate generation."""

    @pytest.mark.asyncio
    async def test_deletion_certificate_generated(
        self, remediation_engine, remediation_config, mock_source_handler, sample_pii_item
    ):
        """Deletion certificate is generated when configured."""
        remediation_config.generate_certificates = True
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)

        # Set as due
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine._execute_action(item_id)

        # Certificate should be generated
        certificate = await remediation_engine.get_deletion_certificate(item_id)
        assert certificate is not None
        assert "deleted_at" in certificate or hasattr(certificate, "deleted_at")

    @pytest.mark.asyncio
    async def test_deletion_certificate_includes_required_fields(
        self, remediation_engine, remediation_config, mock_source_handler, sample_pii_item
    ):
        """Deletion certificate includes all required fields for compliance."""
        remediation_config.generate_certificates = True
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await remediation_engine._execute_action(item_id)

        certificate = await remediation_engine.get_deletion_certificate(item_id)

        # Required fields for GDPR compliance
        required_fields = [
            "item_id",
            "pii_type",
            "source_location",
            "deleted_at",
            "action_taken",
            "certificate_id",
        ]

        for field in required_fields:
            assert hasattr(certificate, field) or field in certificate


# ============================================================================
# TestNotifications
# ============================================================================


class TestRemediationNotifications:
    """Tests for remediation notifications."""

    @pytest.mark.asyncio
    async def test_notification_sent_on_action(
        self, remediation_engine, remediation_config, mock_notification_service, mock_source_handler, sample_pii_item
    ):
        """Notification is sent when remediation action is taken."""
        remediation_config.notify_on_action = True
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine._execute_action(item_id)

        # Notification should be sent
        assert len(mock_notification_service._notifications) > 0

    @pytest.mark.asyncio
    async def test_notification_not_sent_when_disabled(
        self, remediation_engine, remediation_config, mock_notification_service, mock_source_handler, sample_pii_item
    ):
        """Notification is not sent when disabled."""
        remediation_config.notify_on_action = False
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine._execute_action(item_id)

        # Notification should not be sent
        assert len(mock_notification_service._notifications) == 0


# ============================================================================
# TestApprovalWorkflow
# ============================================================================


class TestApprovalWorkflow:
    """Tests for approval workflow."""

    @pytest.mark.asyncio
    async def test_approve_remediation(
        self, remediation_engine, remediation_config, pii_type_enum, test_tenant_id, test_user_id_admin
    ):
        """approve_remediation() marks item as approved."""
        remediation_config.requires_approval = True

        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        # Approve
        await remediation_engine.approve_remediation(item_id, test_user_id_admin)

        item = await remediation_engine.get_item(item_id)
        assert item.status == "approved"
        assert item.approved_by == test_user_id_admin

    @pytest.mark.asyncio
    async def test_cancel_remediation(
        self, remediation_engine, pii_type_enum, test_tenant_id, test_user_id_admin
    ):
        """cancel_remediation() marks item as cancelled."""
        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        # Cancel
        await remediation_engine.cancel_remediation(
            item_id,
            test_user_id_admin,
            reason="False positive",
        )

        item = await remediation_engine.get_item(item_id)
        assert item.status == "cancelled"

    @pytest.mark.asyncio
    async def test_approved_item_processed(
        self, remediation_engine, remediation_config, mock_source_handler, pii_type_enum, test_tenant_id, test_user_id_admin
    ):
        """Approved items are processed."""
        remediation_config.requires_approval = True
        remediation_engine._register_source_handler("database", mock_source_handler)

        item_id = await remediation_engine.schedule_remediation(
            pii_type=pii_type_enum.SSN,
            source_type="database",
            source_location="users.ssn",
            tenant_id=test_tenant_id,
            value_hash="abc123",
        )

        # Set as due
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        # Approve
        await remediation_engine.approve_remediation(item_id, test_user_id_admin)

        # Process
        processed = await remediation_engine.process_pending()

        assert processed >= 1


# ============================================================================
# TestDryRunMode
# ============================================================================


class TestDryRunMode:
    """Tests for dry-run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_logs_without_executing(
        self, remediation_engine_dry_run, mock_source_handler, sample_pii_item
    ):
        """Dry-run mode logs actions without executing."""
        remediation_engine_dry_run._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine_dry_run._create_item(sample_pii_item)
        await remediation_engine_dry_run._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine_dry_run._execute_action(item_id)

        # Source handler should not be called
        mock_source_handler.delete.assert_not_awaited()


# ============================================================================
# TestSourceHandlers
# ============================================================================


class TestSourceHandlers:
    """Tests for source-specific handlers."""

    @pytest.mark.asyncio
    async def test_postgresql_source_handler(
        self, remediation_engine, mock_db_pool
    ):
        """PostgreSQL source handler executes correctly."""
        try:
            from greenlang.infrastructure.pii_service.remediation.handlers import PostgreSQLHandler
            handler = PostgreSQLHandler(mock_db_pool)

            result = await handler.delete(
                location="users.ssn_encrypted",
                value_hash="abc123",
                tenant_id="tenant-123",
            )

            assert result is True
        except ImportError:
            pytest.skip("PostgreSQLHandler not available")

    @pytest.mark.asyncio
    async def test_s3_source_handler(
        self, remediation_engine, mock_s3_client
    ):
        """S3 source handler executes correctly."""
        try:
            from greenlang.infrastructure.pii_service.remediation.handlers import S3Handler
            handler = S3Handler(mock_s3_client)

            result = await handler.delete(
                location="s3://bucket/path/to/file",
                value_hash="abc123",
                tenant_id="tenant-123",
            )

            assert result is True
        except ImportError:
            pytest.skip("S3Handler not available")

    @pytest.mark.asyncio
    async def test_redis_source_handler(
        self, remediation_engine, mock_redis_client
    ):
        """Redis source handler executes correctly."""
        try:
            from greenlang.infrastructure.pii_service.remediation.handlers import RedisHandler
            handler = RedisHandler(mock_redis_client)

            result = await handler.delete(
                location="cache:user:123:ssn",
                value_hash="abc123",
                tenant_id="tenant-123",
            )

            assert result is True
        except ImportError:
            pytest.skip("RedisHandler not available")


# ============================================================================
# TestAuditLogging
# ============================================================================


class TestRemediationAuditLogging:
    """Tests for remediation audit logging."""

    @pytest.mark.asyncio
    async def test_audit_log_on_execution(
        self, remediation_engine, mock_audit_service, mock_source_handler, sample_pii_item
    ):
        """Audit log entry is created on execution."""
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine._execute_action(item_id)

        # Audit log should have entry
        audit_log = mock_audit_service.get_audit_log()
        assert len(audit_log) > 0

    @pytest.mark.asyncio
    async def test_audit_log_includes_details(
        self, remediation_engine, mock_audit_service, mock_source_handler, sample_pii_item
    ):
        """Audit log includes all relevant details."""
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine._execute_action(item_id)

        audit_log = mock_audit_service.get_audit_log()
        # Should include action details
        remediation_events = [e for e in audit_log if "remediation" in str(e).lower()]
        assert len(remediation_events) > 0


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestRemediationErrorHandling:
    """Tests for error handling in remediation."""

    @pytest.mark.asyncio
    async def test_handler_failure_marks_item_failed(
        self, remediation_engine, sample_pii_item
    ):
        """Handler failure marks item as failed."""
        # Handler that fails
        failing_handler = AsyncMock()
        failing_handler.delete = AsyncMock(side_effect=Exception("Delete failed"))
        remediation_engine._register_source_handler("database", failing_handler)

        sample_pii_item["action"] = "delete"
        item_id = await remediation_engine._create_item(sample_pii_item)
        await remediation_engine._update_item_scheduled_time(
            item_id,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )

        await remediation_engine._execute_action(item_id)

        item = await remediation_engine.get_item(item_id)
        assert item.status == "failed"
        assert item.error_message is not None

    @pytest.mark.asyncio
    async def test_retry_failed_items(
        self, remediation_engine, mock_source_handler, sample_pii_item
    ):
        """Failed items can be retried."""
        remediation_engine._register_source_handler("database", mock_source_handler)

        sample_pii_item["action"] = "delete"
        sample_pii_item["status"] = "failed"
        item_id = await remediation_engine._create_item(sample_pii_item)

        # Retry
        await remediation_engine.retry_item(item_id)

        item = await remediation_engine.get_item(item_id)
        assert item.status in ["pending", "completed"]
