"""
Fallback Chain - Agent Factory Resilience (INFRA-010)

Provides an ordered chain of fallback handlers for agent execution.
When the primary handler fails, execution cascades through secondary
handlers until one succeeds or the default handler is reached.

Classes:
    - FallbackHandler: Protocol for fallback handler implementations.
    - FallbackResult: Result of a fallback chain execution.
    - FallbackChain: Ordered chain of fallback handlers.

Example:
    >>> chain = FallbackChain("intake-agent")
    >>> chain.add_handler("primary", primary_handler)
    >>> chain.add_handler("cached", cached_handler)
    >>> chain.set_default(default_error_handler)
    >>> result = await chain.execute(context)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FallbackHandler(Protocol):
    """Protocol for fallback handler implementations.

    Each handler receives a context dictionary and returns a result.
    Handlers should raise an exception to signal failure and trigger
    the next handler in the chain.
    """

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the handler with the given context.

        Args:
            context: Execution context containing input data and metadata.

        Returns:
            The handler result.

        Raises:
            Exception: On failure, to trigger the next fallback.
        """
        ...


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackResult:
    """Result of a fallback chain execution.

    Attributes:
        result: The value returned by the handler that succeeded.
        handler_name: Name of the handler that produced the result.
        was_fallback: True if the primary handler did not succeed.
        duration_ms: Total execution time across all attempted handlers.
        attempted_handlers: Names of all handlers that were attempted.
        errors: Mapping of handler_name -> error message for failed handlers.
    """

    result: Any
    handler_name: str
    was_fallback: bool
    duration_ms: float
    attempted_handlers: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Callable Adapter
# ---------------------------------------------------------------------------


class _CallableAdapter:
    """Adapts an async callable to the FallbackHandler protocol."""

    def __init__(self, fn: Callable[[Dict[str, Any]], Awaitable[Any]]) -> None:
        self._fn = fn

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the wrapped callable."""
        return await self._fn(context)


# ---------------------------------------------------------------------------
# Fallback Chain
# ---------------------------------------------------------------------------


class FallbackChain:
    """Ordered chain of fallback handlers for agent execution.

    Handlers are tried in insertion order. The first handler to succeed
    produces the result. If all named handlers fail, the default handler
    is invoked as a last resort.

    Attributes:
        agent_key: Identifier of the agent this chain protects.
    """

    def __init__(self, agent_key: str) -> None:
        """Initialize a fallback chain.

        Args:
            agent_key: Identifier of the agent this chain serves.
        """
        self.agent_key = agent_key
        self._handlers: List[tuple[str, FallbackHandler]] = []
        self._default: Optional[tuple[str, FallbackHandler]] = None
        self._metrics: Dict[str, _HandlerMetrics] = {}
        logger.debug("FallbackChain created for '%s'", agent_key)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_handler(
        self,
        name: str,
        handler: FallbackHandler | Callable[[Dict[str, Any]], Awaitable[Any]],
    ) -> FallbackChain:
        """Append a named handler to the chain.

        Args:
            name: Human-readable name for the handler.
            handler: Handler instance or async callable.

        Returns:
            Self for fluent chaining.
        """
        if callable(handler) and not isinstance(handler, FallbackHandler):
            handler = _CallableAdapter(handler)
        self._handlers.append((name, handler))
        self._metrics[name] = _HandlerMetrics()
        logger.debug(
            "FallbackChain '%s': added handler '%s' (position %d)",
            self.agent_key, name, len(self._handlers) - 1,
        )
        return self

    def set_default(
        self,
        handler: FallbackHandler | Callable[[Dict[str, Any]], Awaitable[Any]],
        name: str = "default",
    ) -> FallbackChain:
        """Set the default (last-resort) handler.

        Args:
            handler: Handler instance or async callable.
            name: Name for the default handler.

        Returns:
            Self for fluent chaining.
        """
        if callable(handler) and not isinstance(handler, FallbackHandler):
            handler = _CallableAdapter(handler)
        self._default = (name, handler)
        self._metrics[name] = _HandlerMetrics()
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, context: Dict[str, Any]) -> FallbackResult:
        """Execute the fallback chain.

        Tries each handler in order. Returns the first successful result.
        If all handlers fail and a default is set, the default is invoked.

        Args:
            context: Execution context dict.

        Returns:
            FallbackResult with the outcome.

        Raises:
            RuntimeError: If no handlers are configured.
            Exception: If all handlers (including default) fail.
        """
        if not self._handlers and self._default is None:
            raise RuntimeError(
                f"FallbackChain '{self.agent_key}' has no handlers configured."
            )

        start = time.perf_counter()
        attempted: List[str] = []
        errors: Dict[str, str] = {}
        is_primary = True

        for name, handler in self._handlers:
            attempted.append(name)
            handler_start = time.perf_counter()
            try:
                result = await handler.execute(context)
                handler_ms = (time.perf_counter() - handler_start) * 1000
                total_ms = (time.perf_counter() - start) * 1000

                self._record_success(name, handler_ms)
                logger.info(
                    "FallbackChain '%s': handler '%s' succeeded in %.1fms",
                    self.agent_key, name, handler_ms,
                )

                return FallbackResult(
                    result=result,
                    handler_name=name,
                    was_fallback=not is_primary,
                    duration_ms=total_ms,
                    attempted_handlers=list(attempted),
                    errors=dict(errors),
                )
            except Exception as exc:
                handler_ms = (time.perf_counter() - handler_start) * 1000
                self._record_failure(name, handler_ms)
                errors[name] = str(exc)
                logger.warning(
                    "FallbackChain '%s': handler '%s' failed in %.1fms: %s",
                    self.agent_key, name, handler_ms, exc,
                )
                is_primary = False

        # All named handlers failed; try default
        if self._default is not None:
            default_name, default_handler = self._default
            attempted.append(default_name)
            handler_start = time.perf_counter()
            try:
                result = await default_handler.execute(context)
                handler_ms = (time.perf_counter() - handler_start) * 1000
                total_ms = (time.perf_counter() - start) * 1000

                self._record_success(default_name, handler_ms)
                logger.info(
                    "FallbackChain '%s': default handler '%s' succeeded in %.1fms",
                    self.agent_key, default_name, handler_ms,
                )

                return FallbackResult(
                    result=result,
                    handler_name=default_name,
                    was_fallback=True,
                    duration_ms=total_ms,
                    attempted_handlers=list(attempted),
                    errors=dict(errors),
                )
            except Exception as exc:
                handler_ms = (time.perf_counter() - handler_start) * 1000
                self._record_failure(default_name, handler_ms)
                errors[default_name] = str(exc)
                logger.error(
                    "FallbackChain '%s': default handler '%s' ALSO failed: %s",
                    self.agent_key, default_name, exc,
                )

        total_ms = (time.perf_counter() - start) * 1000
        raise RuntimeError(
            f"FallbackChain '{self.agent_key}': all {len(attempted)} handlers failed. "
            f"Errors: {errors}"
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return per-handler metrics.

        Returns:
            Mapping of handler_name -> metrics dict.
        """
        return {
            name: {
                "call_count": m.call_count,
                "success_count": m.success_count,
                "failure_count": m.failure_count,
                "avg_latency_ms": m.total_latency_ms / m.call_count if m.call_count > 0 else 0.0,
            }
            for name, m in self._metrics.items()
        }

    def _record_success(self, name: str, duration_ms: float) -> None:
        """Record a successful handler invocation."""
        if name in self._metrics:
            self._metrics[name].call_count += 1
            self._metrics[name].success_count += 1
            self._metrics[name].total_latency_ms += duration_ms

    def _record_failure(self, name: str, duration_ms: float) -> None:
        """Record a failed handler invocation."""
        if name in self._metrics:
            self._metrics[name].call_count += 1
            self._metrics[name].failure_count += 1
            self._metrics[name].total_latency_ms += duration_ms


# ---------------------------------------------------------------------------
# Internal Metrics
# ---------------------------------------------------------------------------


@dataclass
class _HandlerMetrics:
    """Internal per-handler metrics accumulator."""

    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0


__all__ = [
    "FallbackChain",
    "FallbackHandler",
    "FallbackResult",
]
