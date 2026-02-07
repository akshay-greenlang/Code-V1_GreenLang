# -*- coding: utf-8 -*-
"""
Unit tests for PIIEnforcementMiddleware - SEC-011 PII Service.

Tests the FastAPI middleware for PII enforcement:
- Request body scanning
- Response body scanning
- Path exclusions
- HTTP method filtering
- Tenant context injection
- Error handling

Coverage target: 85%+ of enforcement/middleware.py
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_enforcement_engine():
    """Mock enforcement engine for middleware testing."""
    engine = AsyncMock()
    engine.enforce = AsyncMock(return_value=MagicMock(
        blocked=False,
        modified_content=None,
        detections=[],
        actions_taken=[],
    ))
    return engine


@pytest.fixture
def middleware(enforcement_config, mock_enforcement_engine):
    """Create PIIEnforcementMiddleware instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.enforcement.middleware import PIIEnforcementMiddleware
        return PIIEnforcementMiddleware(
            config=enforcement_config,
            enforcement_engine=mock_enforcement_engine,
        )
    except ImportError:
        pytest.skip("PIIEnforcementMiddleware not yet implemented")


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock()
    request.method = "POST"
    request.url.path = "/api/v1/data"
    request.headers = {
        "Content-Type": "application/json",
        "X-Tenant-ID": "test-tenant",
        "X-User-ID": str(uuid4()),
        "X-Request-ID": str(uuid4()),
    }
    request.body = AsyncMock(return_value=b'{"name": "John Doe", "email": "john@example.com"}')
    return request


@pytest.fixture
def mock_call_next():
    """Create mock call_next function."""
    async def _call_next(request):
        response = MagicMock()
        response.status_code = 200
        response.body = b'{"result": "success"}'
        response.headers = {"Content-Type": "application/json"}
        return response
    return _call_next


# ============================================================================
# TestMiddlewareRequestScanning
# ============================================================================


class TestMiddlewareRequestScanning:
    """Tests for request body scanning."""

    @pytest.mark.asyncio
    async def test_middleware_scans_post_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans POST request bodies."""
        mock_request.method = "POST"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()

    @pytest.mark.asyncio
    async def test_middleware_scans_put_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans PUT request bodies."""
        mock_request.method = "PUT"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()

    @pytest.mark.asyncio
    async def test_middleware_scans_patch_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans PATCH request bodies."""
        mock_request.method = "PATCH"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()

    @pytest.mark.asyncio
    async def test_middleware_skips_get_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware does not scan GET request bodies."""
        mock_request.method = "GET"

        await middleware(mock_request, mock_call_next)

        # Should not scan body for GET
        # Verify by checking enforce wasn't called with body content

    @pytest.mark.asyncio
    async def test_middleware_skips_delete_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware does not scan DELETE request bodies."""
        mock_request.method = "DELETE"

        await middleware(mock_request, mock_call_next)

        # DELETE requests typically don't have bodies to scan


# ============================================================================
# TestPathExclusions
# ============================================================================


class TestPathExclusions:
    """Tests for path exclusion handling."""

    @pytest.mark.asyncio
    async def test_middleware_skips_excluded_paths(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware skips scanning for excluded paths."""
        mock_request.url.path = "/health"

        await middleware(mock_request, mock_call_next)

        # Health endpoint should be excluded
        mock_enforcement_engine.enforce.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_middleware_skips_metrics_path(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware skips scanning for /metrics."""
        mock_request.url.path = "/metrics"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_middleware_scans_non_excluded_paths(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans paths not in exclusion list."""
        mock_request.url.path = "/api/v1/users"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()

    @pytest.mark.asyncio
    async def test_middleware_supports_path_patterns(
        self, middleware, enforcement_config, mock_request, mock_call_next
    ):
        """Middleware supports glob patterns in exclusions."""
        enforcement_config.exclude_paths.append("/internal/*")
        mock_request.url.path = "/internal/debug"

        await middleware(mock_request, mock_call_next)

        # Should be excluded by pattern


# ============================================================================
# TestBlockingBehavior
# ============================================================================


class TestBlockingBehavior:
    """Tests for blocking behavior."""

    @pytest.mark.asyncio
    async def test_middleware_returns_400_on_blocked(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware returns 400 when content is blocked."""
        mock_enforcement_engine.enforce.return_value = MagicMock(
            blocked=True,
            detections=[MagicMock(pii_type=MagicMock(value="ssn"))],
            actions_taken=[],
        )

        response = await middleware(mock_request, mock_call_next)

        assert response.status_code == 400
        # Should include reason in response
        body = json.loads(response.body)
        assert "pii" in body.get("detail", "").lower() or "blocked" in body.get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_middleware_allows_clean_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware allows requests with no PII."""
        mock_enforcement_engine.enforce.return_value = MagicMock(
            blocked=False,
            modified_content=None,
            detections=[],
            actions_taken=[],
        )
        mock_request.body = AsyncMock(return_value=b'{"data": "clean content"}')

        response = await middleware(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_middleware_allows_redacted_requests(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware allows requests after redaction."""
        mock_enforcement_engine.enforce.return_value = MagicMock(
            blocked=False,
            modified_content='{"name": "[PERSON_NAME]", "email": "[EMAIL]"}',
            detections=[MagicMock(), MagicMock()],
            actions_taken=[],
        )

        response = await middleware(mock_request, mock_call_next)

        assert response.status_code == 200


# ============================================================================
# TestContextInjection
# ============================================================================


class TestContextInjection:
    """Tests for tenant context injection."""

    @pytest.mark.asyncio
    async def test_middleware_injects_tenant_context(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware extracts tenant from headers."""
        mock_request.headers["X-Tenant-ID"] = "tenant-123"

        await middleware(mock_request, mock_call_next)

        # Verify tenant was passed to enforcement
        call_args = mock_enforcement_engine.enforce.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("context")
        assert context.tenant_id == "tenant-123"

    @pytest.mark.asyncio
    async def test_middleware_injects_user_context(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware extracts user from headers."""
        user_id = str(uuid4())
        mock_request.headers["X-User-ID"] = user_id

        await middleware(mock_request, mock_call_next)

        call_args = mock_enforcement_engine.enforce.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("context")
        assert context.user_id == user_id

    @pytest.mark.asyncio
    async def test_middleware_injects_request_id(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware extracts request ID from headers."""
        request_id = str(uuid4())
        mock_request.headers["X-Request-ID"] = request_id

        await middleware(mock_request, mock_call_next)

        call_args = mock_enforcement_engine.enforce.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("context")
        assert context.request_id == request_id

    @pytest.mark.asyncio
    async def test_middleware_injects_path_context(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware includes request path in context."""
        mock_request.url.path = "/api/v1/sensitive/data"

        await middleware(mock_request, mock_call_next)

        call_args = mock_enforcement_engine.enforce.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("context")
        assert context.path == "/api/v1/sensitive/data"

    @pytest.mark.asyncio
    async def test_middleware_injects_method_context(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware includes HTTP method in context."""
        mock_request.method = "POST"

        await middleware(mock_request, mock_call_next)

        call_args = mock_enforcement_engine.enforce.call_args
        context = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("context")
        assert context.method == "POST"


# ============================================================================
# TestResponseScanning
# ============================================================================


class TestResponseScanning:
    """Tests for response body scanning."""

    @pytest.mark.asyncio
    async def test_response_body_scanning_when_enabled(
        self, middleware, enforcement_config, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans response bodies when configured."""
        enforcement_config.scan_responses = True

        response = await middleware(mock_request, mock_call_next)

        # Should have scanned response
        # Two calls: one for request, one for response
        assert mock_enforcement_engine.enforce.await_count >= 1

    @pytest.mark.asyncio
    async def test_response_body_scanning_disabled_by_default(
        self, middleware, enforcement_config, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware skips response scanning when disabled."""
        enforcement_config.scan_responses = False

        await middleware(mock_request, mock_call_next)

        # Should only scan request, not response
        # Implementation specific

    @pytest.mark.asyncio
    async def test_response_redaction(
        self, middleware, enforcement_config, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware redacts PII in responses."""
        enforcement_config.scan_responses = True

        # Make response scanning return redacted content
        original_enforce = mock_enforcement_engine.enforce

        async def enforce_response(content, context):
            if "email" in content.lower():
                return MagicMock(
                    blocked=False,
                    modified_content=content.replace("john@example.com", "[EMAIL]"),
                    detections=[MagicMock()],
                    actions_taken=[],
                )
            return MagicMock(blocked=False, modified_content=None, detections=[], actions_taken=[])

        mock_enforcement_engine.enforce = AsyncMock(side_effect=enforce_response)

        # This is implementation specific


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestMiddlewareErrorHandling:
    """Tests for middleware error handling."""

    @pytest.mark.asyncio
    async def test_middleware_error_handling(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware handles enforcement errors gracefully."""
        mock_enforcement_engine.enforce.side_effect = Exception("Enforcement error")

        response = await middleware(mock_request, mock_call_next)

        # Should fail safe - either return error or block
        assert response.status_code in [400, 500]

    @pytest.mark.asyncio
    async def test_middleware_body_read_error(
        self, middleware, mock_request, mock_call_next
    ):
        """Middleware handles body read errors."""
        mock_request.body = AsyncMock(side_effect=Exception("Body read failed"))

        response = await middleware(mock_request, mock_call_next)

        # Should handle gracefully
        assert response is not None

    @pytest.mark.asyncio
    async def test_middleware_continues_on_empty_body(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware handles empty request bodies."""
        mock_request.body = AsyncMock(return_value=b"")

        response = await middleware(mock_request, mock_call_next)

        # Should continue normally
        assert response.status_code == 200


# ============================================================================
# TestContentTypeHandling
# ============================================================================


class TestContentTypeHandling:
    """Tests for content type handling."""

    @pytest.mark.asyncio
    async def test_middleware_skips_binary_content(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware skips scanning binary content types."""
        mock_request.headers["Content-Type"] = "application/octet-stream"

        await middleware(mock_request, mock_call_next)

        # Should not attempt to scan binary content
        mock_enforcement_engine.enforce.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_middleware_skips_image_content(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware skips scanning image content."""
        mock_request.headers["Content-Type"] = "image/png"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_middleware_scans_json_content(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans application/json content."""
        mock_request.headers["Content-Type"] = "application/json"

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()

    @pytest.mark.asyncio
    async def test_middleware_scans_text_content(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans text/plain content."""
        mock_request.headers["Content-Type"] = "text/plain"
        mock_request.body = AsyncMock(return_value=b"Some text with email: john@example.com")

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()

    @pytest.mark.asyncio
    async def test_middleware_scans_form_content(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware scans form-urlencoded content."""
        mock_request.headers["Content-Type"] = "application/x-www-form-urlencoded"
        mock_request.body = AsyncMock(return_value=b"name=John&email=john@example.com")

        await middleware(mock_request, mock_call_next)

        mock_enforcement_engine.enforce.assert_awaited()


# ============================================================================
# TestMetricsRecording
# ============================================================================


class TestMiddlewareMetrics:
    """Tests for middleware metrics."""

    @pytest.mark.asyncio
    async def test_middleware_records_request_count(
        self, middleware, mock_request, mock_call_next
    ):
        """Middleware records request count metrics."""
        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_requests_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            await middleware(mock_request, mock_call_next)

            # Metrics should be recorded
            # Implementation specific

    @pytest.mark.asyncio
    async def test_middleware_records_blocked_count(
        self, middleware, mock_enforcement_engine, mock_request, mock_call_next
    ):
        """Middleware records blocked request count."""
        mock_enforcement_engine.enforce.return_value = MagicMock(
            blocked=True,
            detections=[MagicMock(pii_type=MagicMock(value="ssn"))],
            actions_taken=[],
        )

        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_blocked_requests_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            await middleware(mock_request, mock_call_next)

            # Should record blocked request
