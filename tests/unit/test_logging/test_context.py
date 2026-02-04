# -*- coding: utf-8 -*-
"""
Unit Tests for Logging Context Management - INFRA-009

Tests the context-variable helpers: bind_context, clear_context, get_context,
logging_context (context manager), convenience binders (bind_agent_context,
bind_request_context), thread isolation, and integration with structlog output.

Module under test: greenlang.infrastructure.logging.context
"""

import io
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import structlog

from greenlang.infrastructure.logging.config import LoggingConfig, reset_config
from greenlang.infrastructure.logging.context import (
    bind_agent_context,
    bind_context,
    bind_request_context,
    clear_context,
    get_context,
    logging_context,
)
from greenlang.infrastructure.logging.setup import (
    configure_logging,
    get_logger,
    reset_logging,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure context is cleared before and after every test."""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def _configure_json_logging():
    """Set up structlog with JSON output for integration tests."""
    config = LoggingConfig(
        level="DEBUG",
        format="json",
        service_name="ctx-test",
        environment="test",
        enable_redaction=False,
        async_logging=False,
    )
    configure_logging(config)
    yield
    reset_logging()
    reset_config()
    structlog.reset_defaults()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# bind_context / clear_context / get_context
# ---------------------------------------------------------------------------


class TestBindContext:
    """Tests for bind_context and get_context."""

    def test_bind_context(self):
        """bind_context stores key-value pairs retrievable via get_context."""
        bind_context(tenant_id="acme", request_id="req-001")
        ctx = get_context()
        assert ctx["tenant_id"] == "acme"
        assert ctx["request_id"] == "req-001"

    def test_bind_context_accumulates(self):
        """Multiple bind_context calls accumulate (merge) values."""
        bind_context(a="1")
        bind_context(b="2")
        ctx = get_context()
        assert ctx["a"] == "1"
        assert ctx["b"] == "2"

    def test_bind_context_overwrites(self):
        """Binding the same key again overwrites the previous value."""
        bind_context(x="old")
        bind_context(x="new")
        assert get_context()["x"] == "new"

    def test_bind_context_with_none_values(self):
        """None values are accepted gracefully (stored, no crash)."""
        bind_context(maybe=None)
        ctx = get_context()
        # structlog.contextvars stores None values
        assert ctx.get("maybe") is None


class TestClearContext:
    """Tests for clear_context."""

    def test_clear_context(self):
        """clear_context removes all previously bound variables."""
        bind_context(a="1", b="2")
        clear_context()
        ctx = get_context()
        assert "a" not in ctx
        assert "b" not in ctx

    def test_clear_context_idempotent(self):
        """Calling clear_context on an empty context does not raise."""
        clear_context()
        clear_context()  # Should not raise


class TestGetContext:
    """Tests for get_context."""

    def test_get_context_empty(self):
        """get_context returns empty dict when nothing is bound."""
        ctx = get_context()
        assert isinstance(ctx, dict)


# ---------------------------------------------------------------------------
# logging_context (context manager)
# ---------------------------------------------------------------------------


class TestLoggingContext:
    """Tests for the logging_context context manager."""

    def test_logging_context_manager(self):
        """Context manager binds on enter and restores on exit."""
        bind_context(outer="before")

        with logging_context(inner="during"):
            ctx_inside = get_context()
            assert ctx_inside.get("inner") == "during"
            # outer should still be visible
            assert ctx_inside.get("outer") == "before"

        ctx_after = get_context()
        # inner should be gone after exit
        assert "inner" not in ctx_after
        # outer should be restored
        assert ctx_after.get("outer") == "before"

    def test_nested_logging_context(self):
        """Nested context managers do not leak inner bindings to outer scopes."""
        with logging_context(level="L1"):
            assert get_context().get("level") == "L1"

            with logging_context(level="L2", extra="nested"):
                assert get_context().get("level") == "L2"
                assert get_context().get("extra") == "nested"

            # Back to L1
            ctx_l1 = get_context()
            assert ctx_l1.get("level") == "L1"
            assert "extra" not in ctx_l1

        # Fully exited
        assert "level" not in get_context()

    def test_logging_context_exception_safety(self):
        """Context is properly restored even when an exception occurs."""
        bind_context(safe="yes")

        with pytest.raises(RuntimeError):
            with logging_context(unsafe="maybe"):
                raise RuntimeError("boom")

        ctx = get_context()
        assert "unsafe" not in ctx
        assert ctx.get("safe") == "yes"


# ---------------------------------------------------------------------------
# Convenience Binders
# ---------------------------------------------------------------------------


class TestConvenienceBinders:
    """Tests for bind_agent_context and bind_request_context."""

    def test_bind_agent_context(self):
        """bind_agent_context sets agent_name, agent_id, and execution_id."""
        bind_agent_context(
            agent_name="EmissionCalcAgent",
            agent_id="agent-001",
            execution_id="exec-abc",
        )
        ctx = get_context()
        assert ctx["agent_name"] == "EmissionCalcAgent"
        assert ctx["agent_id"] == "agent-001"
        assert ctx["execution_id"] == "exec-abc"

    def test_bind_request_context(self):
        """bind_request_context sets request_id, tenant_id, and user_id."""
        bind_request_context(
            request_id="req-xyz",
            tenant_id="tenant-acme",
            user_id="user-42",
        )
        ctx = get_context()
        assert ctx["request_id"] == "req-xyz"
        assert ctx["tenant_id"] == "tenant-acme"
        assert ctx["user_id"] == "user-42"

    def test_bind_request_context_optional_fields(self):
        """bind_request_context works when tenant_id and user_id are None."""
        bind_request_context(request_id="req-minimal")
        ctx = get_context()
        assert ctx["request_id"] == "req-minimal"
        assert "tenant_id" not in ctx
        assert "user_id" not in ctx


# ---------------------------------------------------------------------------
# Thread Isolation
# ---------------------------------------------------------------------------


class TestThreadIsolation:
    """Tests verifying context variables are isolated between threads.

    structlog uses Python contextvars which provide per-task/per-thread
    isolation. Each thread gets its own copy of the context.
    """

    def test_context_thread_isolation(self):
        """Context bound in one thread is not visible in another thread."""
        barrier = threading.Barrier(2, timeout=5)
        results = {}

        def thread_a():
            clear_context()
            bind_context(thread="A", value="alpha")
            barrier.wait()  # Sync with thread B
            ctx = get_context()
            results["A"] = dict(ctx)

        def thread_b():
            clear_context()
            bind_context(thread="B", value="beta")
            barrier.wait()  # Sync with thread A
            ctx = get_context()
            results["B"] = dict(ctx)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(thread_a), pool.submit(thread_b)]
            for f in as_completed(futures):
                f.result()  # re-raise if any thread errored

        # Thread A should only see its own context
        assert results["A"].get("thread") == "A"
        assert results["A"].get("value") == "alpha"

        # Thread B should only see its own context
        assert results["B"].get("thread") == "B"
        assert results["B"].get("value") == "beta"


# ---------------------------------------------------------------------------
# Integration with structlog Output
# ---------------------------------------------------------------------------


class TestContextInLogOutput:
    """Tests verifying bound context appears in structlog output."""

    def test_context_appears_in_log_output(self, _configure_json_logging):
        """Variables bound via bind_context are included in the JSON log line.

        structlog's merge_contextvars processor injects bound context vars
        into every event dict before rendering.
        """
        bind_context(correlation_id="corr-12345", tenant_id="acme")

        logger = get_logger("test.ctx_output")

        # Capture structlog output (writes to stdout via PrintLoggerFactory)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            logger.info("context_output_test")
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue().strip()

        parsed = json.loads(output)
        assert parsed.get("correlation_id") == "corr-12345", (
            f"correlation_id not found in log output: {parsed}"
        )
        assert parsed.get("tenant_id") == "acme", (
            f"tenant_id not found in log output: {parsed}"
        )
