"""
Context Variable Management - INFRA-009

Provides convenience functions for binding, clearing, and retrieving
structured logging context variables (correlation IDs, tenant IDs, agent
execution metadata, etc.) using :mod:`structlog.contextvars` as the
underlying storage mechanism.

Context variables are thread-safe and async-safe (they use Python
:mod:`contextvars` under the hood). Values bound with :func:`bind_context`
are automatically merged into every subsequent log entry by the
``merge_contextvars`` processor in the structlog chain.

Functions:
    - bind_context: Bind arbitrary key-value pairs into the current context.
    - clear_context: Clear all bound context variables.
    - get_context: Retrieve the currently bound context as a dict.
    - logging_context: Context manager that binds on enter and restores on exit.
    - bind_agent_context: Convenience for agent execution logging.
    - bind_request_context: Convenience for HTTP request logging.

Example:
    >>> from greenlang.infrastructure.logging.context import (
    ...     bind_context, get_context, clear_context, logging_context,
    ... )
    >>> bind_context(request_id="req-123", tenant_id="t-acme")
    >>> ctx = get_context()
    >>> ctx["request_id"]
    'req-123'
    >>> clear_context()
    >>> get_context()
    {}
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

import structlog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs into the current structured logging context.

    These values will be automatically included in every subsequent log
    entry until :func:`clear_context` is called or the context exits
    scope (for async tasks using contextvars copy semantics).

    Common fields include ``request_id``, ``trace_id``, ``span_id``,
    ``tenant_id``, ``user_id``, ``operation``, and ``agent_name``.

    Args:
        **kwargs: Arbitrary key-value pairs to bind. Values should be
            JSON-serializable for structured output.

    Example:
        >>> bind_context(request_id="req-abc", tenant_id="t-acme")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables from the current context.

    After calling this, log entries will no longer include previously
    bound context fields (until new ones are bound).

    Example:
        >>> bind_context(request_id="req-abc")
        >>> clear_context()
        >>> get_context()
        {}
    """
    structlog.contextvars.clear_contextvars()


def get_context() -> dict[str, Any]:
    """Retrieve the currently bound context variables as a dictionary.

    Returns:
        A dictionary of all currently bound context variables. Returns an
        empty dict if no context has been bound.

    Example:
        >>> bind_context(request_id="req-abc", tenant_id="t-acme")
        >>> ctx = get_context()
        >>> ctx["request_id"]
        'req-abc'
    """
    return structlog.contextvars.get_contextvars()


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


@contextmanager
def logging_context(**kwargs: Any) -> Generator[dict[str, Any], None, None]:
    """Context manager that binds context on entry and restores on exit.

    Saves the current context before entering, binds the provided
    key-value pairs, and restores the original context on exit. This
    is useful for scoped operations like handling a single request or
    running a sub-pipeline where you want temporary context.

    Args:
        **kwargs: Key-value pairs to bind for the duration of the block.

    Yields:
        The merged context dictionary (original + new bindings).

    Example:
        >>> with logging_context(operation="calculate_emissions") as ctx:
        ...     # All logs here include operation="calculate_emissions"
        ...     pass
        >>> # After the block, the original context is restored
    """
    # Save current context
    previous_context = get_context()

    # Bind new context (merges with existing)
    bind_context(**kwargs)

    try:
        yield get_context()
    finally:
        # Restore previous context by clearing and re-binding
        clear_context()
        if previous_context:
            bind_context(**previous_context)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def bind_agent_context(
    agent_name: str,
    agent_id: str,
    execution_id: str,
) -> None:
    """Bind agent execution metadata into the logging context.

    This is a convenience wrapper for the common pattern of logging
    agent pipeline executions. Binds ``agent_name``, ``agent_id``, and
    ``execution_id`` so that every log entry from the agent includes
    these fields for traceability.

    Args:
        agent_name: The name/type of the agent (e.g. "EmissionCalculationAgent").
        agent_id: Unique identifier of the agent instance.
        execution_id: Unique identifier of this execution run.

    Example:
        >>> bind_agent_context(
        ...     agent_name="EmissionCalculationAgent",
        ...     agent_id="agent-001",
        ...     execution_id="exec-abc-123",
        ... )
    """
    bind_context(
        agent_name=agent_name,
        agent_id=agent_id,
        execution_id=execution_id,
    )


def bind_request_context(
    request_id: str,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Bind HTTP request metadata into the logging context.

    This is a convenience wrapper for the common pattern of logging
    HTTP request handling. Binds ``request_id`` and optionally
    ``tenant_id`` and ``user_id``.

    Args:
        request_id: Unique identifier for the HTTP request.
        tenant_id: Optional tenant identifier for multi-tenant isolation.
        user_id: Optional user identifier.

    Example:
        >>> bind_request_context(
        ...     request_id="req-456",
        ...     tenant_id="t-acme",
        ...     user_id="u-jane",
        ... )
    """
    context: dict[str, Any] = {"request_id": request_id}

    if tenant_id is not None:
        context["tenant_id"] = tenant_id

    if user_id is not None:
        context["user_id"] = user_id

    bind_context(**context)
