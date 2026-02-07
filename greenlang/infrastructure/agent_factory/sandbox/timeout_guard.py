"""
Sandbox Timeout Guard - Agent Factory Sandbox (INFRA-010)

Wraps subprocess execution with timeout enforcement. Implements
graceful termination (SIGTERM) followed by forced kill (SIGKILL)
after a grace period. Reports timeout events for auditing.

Classes:
    - SandboxTimeoutGuard: Timeout enforcement for sandboxed subprocesses.
    - SandboxTimeoutEvent: Dataclass representing a timeout occurrence.

Example:
    >>> guard = SandboxTimeoutGuard(timeout_s=60.0, grace_period_s=5.0)
    >>> stdout, stderr, exit_code = await guard.run_with_timeout(process)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeout Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxTimeoutEvent:
    """Record of a sandbox timeout occurrence.

    Attributes:
        agent_key: The agent that timed out.
        timeout_s: The configured timeout in seconds.
        elapsed_s: Actual elapsed time in seconds.
        graceful_kill: Whether SIGTERM was sent before SIGKILL.
        forced_kill: Whether SIGKILL was required.
        timestamp: When the timeout occurred (UTC).
    """

    agent_key: str
    timeout_s: float
    elapsed_s: float
    graceful_kill: bool
    forced_kill: bool
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Sandbox Timeout Guard
# ---------------------------------------------------------------------------


class SandboxTimeoutGuard:
    """Timeout enforcement for sandboxed subprocess execution.

    Implements a two-phase termination strategy:
    1. SIGTERM with a configurable grace period.
    2. SIGKILL if the process does not terminate within the grace period.

    Records timeout events for downstream auditing and alerting.

    Attributes:
        timeout_s: Maximum execution time in seconds.
        grace_period_s: Seconds to wait between SIGTERM and SIGKILL.
        agent_key: Optional agent identifier for event reporting.
    """

    def __init__(
        self,
        timeout_s: float = 60.0,
        grace_period_s: float = 5.0,
        agent_key: str = "",
    ) -> None:
        """Initialize the timeout guard.

        Args:
            timeout_s: Maximum execution time in seconds.
            grace_period_s: Grace period between SIGTERM and SIGKILL.
            agent_key: Agent identifier for event reporting.
        """
        self.timeout_s = timeout_s
        self.grace_period_s = grace_period_s
        self.agent_key = agent_key
        self._events: List[SandboxTimeoutEvent] = []
        self._on_timeout: Optional[
            Callable[[SandboxTimeoutEvent], Any]
        ] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def on_timeout(
        self,
        callback: Callable[[SandboxTimeoutEvent], Any],
    ) -> SandboxTimeoutGuard:
        """Register a callback for timeout events.

        Args:
            callback: Function or coroutine called with a SandboxTimeoutEvent.

        Returns:
            Self for fluent chaining.
        """
        self._on_timeout = callback
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_with_timeout(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[str, str, int]:
        """Wait for a subprocess to complete within the timeout.

        If the process exceeds the timeout, it is terminated gracefully
        (SIGTERM) and then forcibly (SIGKILL) if necessary.

        Args:
            process: An asyncio subprocess process.

        Returns:
            Tuple of (stdout, stderr, exit_code). Exit code is -1 for timeouts.
        """
        start = time.perf_counter()

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_s,
            )
            return (
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
                process.returncode or 0,
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            logger.warning(
                "SandboxTimeoutGuard: process exceeded %.1fs timeout "
                "(elapsed: %.1fs, agent: %s)",
                self.timeout_s, elapsed, self.agent_key,
            )

            graceful = False
            forced = False

            # Phase 1: SIGTERM
            try:
                process.terminate()
                graceful = True
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self.grace_period_s,
                    )
                except asyncio.TimeoutError:
                    # Phase 2: SIGKILL
                    logger.warning(
                        "SandboxTimeoutGuard: SIGTERM grace period expired, "
                        "sending SIGKILL (agent: %s)",
                        self.agent_key,
                    )
                    process.kill()
                    forced = True
                    await process.wait()
            except ProcessLookupError:
                logger.debug("Process already terminated")

            # Record the timeout event
            event = SandboxTimeoutEvent(
                agent_key=self.agent_key,
                timeout_s=self.timeout_s,
                elapsed_s=elapsed,
                graceful_kill=graceful,
                forced_kill=forced,
            )
            self._events.append(event)
            await self._emit_timeout_event(event)

            return (
                "",
                f"TIMEOUT: execution exceeded {self.timeout_s:.0f}s limit",
                -1,
            )

    # ------------------------------------------------------------------
    # Event History
    # ------------------------------------------------------------------

    @property
    def events(self) -> List[SandboxTimeoutEvent]:
        """Return the list of timeout events recorded by this guard."""
        return list(self._events)

    @property
    def timeout_count(self) -> int:
        """Return the number of timeout events."""
        return len(self._events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _emit_timeout_event(self, event: SandboxTimeoutEvent) -> None:
        """Invoke the timeout callback if registered."""
        if self._on_timeout is None:
            return
        try:
            result = self._on_timeout(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.error(
                "SandboxTimeoutGuard: timeout callback failed: %s", exc,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot.

        Returns:
            Dictionary with configuration and event counts.
        """
        return {
            "agent_key": self.agent_key,
            "timeout_s": self.timeout_s,
            "grace_period_s": self.grace_period_s,
            "timeout_count": self.timeout_count,
            "last_event": (
                {
                    "elapsed_s": self._events[-1].elapsed_s,
                    "forced_kill": self._events[-1].forced_kill,
                    "timestamp": self._events[-1].timestamp.isoformat(),
                }
                if self._events
                else None
            ),
        }


__all__ = [
    "SandboxTimeoutEvent",
    "SandboxTimeoutGuard",
]
