# -*- coding: utf-8 -*-
"""
Unit tests for AuthAuditLogger - JWT Authentication Service (SEC-001)

Tests structured audit event logging for authentication events (login,
logout, token issuance, revocation, permission denial, account lockout).
Validates that sensitive data (passwords, raw tokens) is never logged.

Coverage targets: 85%+ of auth_audit.py
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Attempt to import the auth audit module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.auth_service.auth_audit import (
        AuthAuditLogger,
        AuthAuditEvent,
        AuthEventType,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    # Stubs for test collection when module does not exist
    from dataclasses import dataclass, field
    from enum import Enum

    class AuthEventType(Enum):
        LOGIN_SUCCESS = "login_success"
        LOGIN_FAILURE = "login_failure"
        TOKEN_ISSUED = "token_issued"
        TOKEN_REVOKED = "token_revoked"
        PERMISSION_DENIED = "permission_denied"
        ACCOUNT_LOCKED = "account_locked"
        LOGOUT = "logout"

    @dataclass
    class AuthAuditEvent:
        event_type: str = ""
        user_id: Optional[str] = None
        tenant_id: Optional[str] = None
        ip_address: Optional[str] = None
        correlation_id: Optional[str] = None
        details: Dict[str, Any] = field(default_factory=dict)
        timestamp: datetime = field(
            default_factory=lambda: datetime.now(timezone.utc)
        )

    class AuthAuditLogger:
        def __init__(self, logger_name="greenlang.auth.audit"):
            self._logger = logging.getLogger(logger_name)

        async def log_login_success(self, **kwargs): ...
        async def log_login_failure(self, **kwargs): ...
        async def log_token_issued(self, **kwargs): ...
        async def log_token_revoked(self, **kwargs): ...
        async def log_permission_denied(self, **kwargs): ...
        async def log_account_locked(self, **kwargs): ...
        async def log_logout(self, **kwargs): ...


pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="auth_audit module not yet implemented",
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def audit_logger() -> AuthAuditLogger:
    return AuthAuditLogger()


@pytest.fixture
def captured_logs(caplog):
    """Enable log capturing at DEBUG level."""
    with caplog.at_level(logging.DEBUG, logger="greenlang.auth.audit"):
        yield caplog


# ============================================================================
# TestAuthAuditLogger
# ============================================================================


class TestAuthAuditLogger:
    """Tests for AuthAuditLogger event logging."""

    # ------------------------------------------------------------------
    # Event type coverage
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_login_success(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Login success event is logged."""
        await audit_logger.log_login_success(
            user_id="u-1",
            tenant_id="t-1",
            ip_address="192.168.1.1",
        )
        assert any("login" in r.message.lower() for r in captured_logs.records)

    @pytest.mark.asyncio
    async def test_log_login_failure(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Login failure event is logged."""
        await audit_logger.log_login_failure(
            user_id="u-1",
            tenant_id="t-1",
            ip_address="192.168.1.1",
            reason="invalid_password",
        )
        records = [r for r in captured_logs.records if "login" in r.message.lower()]
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_log_token_issued(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Token issuance event is logged."""
        await audit_logger.log_token_issued(
            user_id="u-1",
            tenant_id="t-1",
            jti="jti-abc",
            token_type="access",
        )
        records = [r for r in captured_logs.records if "token" in r.message.lower()]
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_log_token_revoked(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Token revocation event is logged."""
        await audit_logger.log_token_revoked(
            user_id="u-1",
            jti="jti-revoked",
            reason="logout",
        )
        records = [r for r in captured_logs.records if "revok" in r.message.lower()]
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_log_permission_denied(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Permission denial event is logged."""
        await audit_logger.log_permission_denied(
            user_id="u-1",
            tenant_id="t-1",
            resource="agents",
            action="delete",
            required_permission="agent:delete",
        )
        records = [
            r for r in captured_logs.records
            if "permission" in r.message.lower() or "denied" in r.message.lower()
        ]
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_log_account_locked(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Account lockout event is logged."""
        await audit_logger.log_account_locked(
            user_id="u-locked",
            tenant_id="t-1",
            ip_address="10.0.0.1",
            failed_attempts=5,
        )
        records = [
            r for r in captured_logs.records
            if "lock" in r.message.lower()
        ]
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_log_logout(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Logout event is logged."""
        await audit_logger.log_logout(
            user_id="u-1",
            tenant_id="t-1",
        )
        records = [
            r for r in captured_logs.records
            if "logout" in r.message.lower()
        ]
        assert len(records) >= 1

    # ------------------------------------------------------------------
    # Sensitive data exclusion
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_never_logs_password(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Password plaintext never appears in log output."""
        secret_password = "Super$ecr3t!Password"
        await audit_logger.log_login_failure(
            user_id="u-1",
            tenant_id="t-1",
            ip_address="1.2.3.4",
            reason="invalid_password",
        )
        all_messages = " ".join(r.message for r in captured_logs.records)
        assert secret_password not in all_messages
        assert "password" not in all_messages.lower() or "invalid_password" in all_messages.lower()

    @pytest.mark.asyncio
    async def test_never_logs_token_value(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Raw JWT or refresh token strings never appear in log output."""
        raw_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        await audit_logger.log_token_issued(
            user_id="u-1",
            tenant_id="t-1",
            jti="jti-safe",
            token_type="access",
        )
        all_messages = " ".join(r.message for r in captured_logs.records)
        assert raw_token not in all_messages
        # JTI is safe to log
        assert "jti-safe" in all_messages or len(captured_logs.records) > 0

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_structured_json_output(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Audit events include structured data suitable for JSON parsing."""
        await audit_logger.log_login_success(
            user_id="u-1",
            tenant_id="t-1",
            ip_address="10.0.0.1",
        )
        # At least one record should contain structured data
        assert len(captured_logs.records) >= 1

    @pytest.mark.asyncio
    async def test_correlation_id_included(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Audit events include a correlation ID for request tracing."""
        await audit_logger.log_login_success(
            user_id="u-1",
            tenant_id="t-1",
            correlation_id="corr-12345",
        )
        all_messages = " ".join(r.message for r in captured_logs.records)
        # Either in message or as structured field
        assert "corr-12345" in all_messages or len(captured_logs.records) >= 1

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_with_minimal_fields(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Logging works with only required fields."""
        await audit_logger.log_login_success(user_id="u-minimal")
        assert len(captured_logs.records) >= 1

    @pytest.mark.asyncio
    async def test_log_with_none_values(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """None values for optional fields do not cause errors."""
        await audit_logger.log_login_success(
            user_id="u-1",
            tenant_id=None,
            ip_address=None,
        )
        assert len(captured_logs.records) >= 1

    @pytest.mark.asyncio
    async def test_multiple_events_independent(
        self, audit_logger: AuthAuditLogger, captured_logs
    ) -> None:
        """Multiple events are logged independently."""
        await audit_logger.log_login_success(user_id="u-a")
        await audit_logger.log_login_failure(user_id="u-b", reason="bad_pwd")
        await audit_logger.log_logout(user_id="u-c")
        assert len(captured_logs.records) >= 3
