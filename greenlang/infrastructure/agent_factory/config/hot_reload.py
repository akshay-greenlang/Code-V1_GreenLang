"""
Config Hot Reload - Agent Factory Config (INFRA-010)

Provides live configuration reloading via Redis pub/sub. When an agent
config is updated in the ConfigStore, a change notification is published.
The ConfigHotReload listener receives the notification and invokes
registered callbacks so running agents can pick up new settings without
restarts.

Classes:
    - ConfigChangeEvent: Dataclass representing a configuration change.
    - ConfigHotReload: Redis pub/sub listener for config change events.

Example:
    >>> reload = ConfigHotReload(redis_client)
    >>> reload.register("intake-agent", on_config_change)
    >>> await reload.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Change Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigChangeEvent:
    """Represents a configuration change notification.

    Attributes:
        agent_key: The agent whose config changed.
        old_config: Previous configuration (may be empty on first set).
        new_config: New configuration.
        changed_keys: Set of field names that changed.
        config_version: Version number of the new configuration.
        changed_by: Identity of who made the change.
        timestamp: When the change occurred (UTC).
    """

    agent_key: str
    old_config: Dict[str, Any] = field(default_factory=dict)
    new_config: Dict[str, Any] = field(default_factory=dict)
    changed_keys: frozenset[str] = field(default_factory=frozenset)
    config_version: int = 0
    changed_by: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Callback Type
# ---------------------------------------------------------------------------

ReloadCallback = Callable[[ConfigChangeEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Config Hot Reload
# ---------------------------------------------------------------------------


class ConfigHotReload:
    """Redis pub/sub listener for live configuration changes.

    Subscribes to a Redis channel for config change notifications.
    When a change is received, invokes registered callbacks for the
    affected agent(s). Supports per-agent and wildcard callbacks.

    Implements distributed config versioning to prevent stale overwrites.

    Attributes:
        channel: Redis pub/sub channel for config notifications.
    """

    _DEFAULT_CHANNEL = "gl:config:changes"

    def __init__(
        self,
        redis_client: Any,
        channel: Optional[str] = None,
    ) -> None:
        """Initialize the hot reload listener.

        Args:
            redis_client: Async Redis client (redis.asyncio).
            channel: Redis pub/sub channel. Defaults to 'gl:config:changes'.
        """
        self._redis = redis_client
        self.channel = channel or self._DEFAULT_CHANNEL
        self._callbacks: Dict[str, List[ReloadCallback]] = {}
        self._global_callbacks: List[ReloadCallback] = []
        self._pubsub: Optional[Any] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        self._known_versions: Dict[str, int] = {}
        logger.info(
            "ConfigHotReload initialised (channel=%s)", self.channel,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        agent_key: str,
        callback: ReloadCallback,
    ) -> None:
        """Register a callback for a specific agent's config changes.

        Args:
            agent_key: The agent to watch.
            callback: Async function invoked with a ConfigChangeEvent.
        """
        if agent_key not in self._callbacks:
            self._callbacks[agent_key] = []
        self._callbacks[agent_key].append(callback)
        logger.debug(
            "ConfigHotReload: registered callback for '%s' (total=%d)",
            agent_key, len(self._callbacks[agent_key]),
        )

    def register_global(self, callback: ReloadCallback) -> None:
        """Register a callback for all config changes.

        Args:
            callback: Async function invoked with a ConfigChangeEvent.
        """
        self._global_callbacks.append(callback)
        logger.debug(
            "ConfigHotReload: registered global callback (total=%d)",
            len(self._global_callbacks),
        )

    def unregister(self, agent_key: str) -> int:
        """Remove all callbacks for a specific agent.

        Args:
            agent_key: The agent to stop watching.

        Returns:
            Number of callbacks removed.
        """
        callbacks = self._callbacks.pop(agent_key, [])
        return len(callbacks)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background pub/sub listener task.

        Subscribes to the Redis channel and begins processing messages.
        """
        if self._running:
            logger.debug("ConfigHotReload: already running")
            return

        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(self.channel)
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info(
            "ConfigHotReload: started listening on '%s'", self.channel,
        )

    async def stop(self) -> None:
        """Stop the background listener and clean up."""
        self._running = False
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub is not None:
            await self._pubsub.unsubscribe(self.channel)
            await self._pubsub.close()
            self._pubsub = None
        logger.info("ConfigHotReload: stopped")

    @property
    def is_running(self) -> bool:
        """Whether the listener is currently running."""
        return self._running

    # ------------------------------------------------------------------
    # Publishing (used by ConfigStore)
    # ------------------------------------------------------------------

    async def publish_change(self, event: ConfigChangeEvent) -> int:
        """Publish a config change notification to Redis.

        Args:
            event: The change event to broadcast.

        Returns:
            Number of subscribers that received the message.
        """
        # Track version to prevent stale overwrites
        current_version = self._known_versions.get(event.agent_key, 0)
        if event.config_version <= current_version:
            logger.warning(
                "ConfigHotReload: skipping stale publish for '%s' "
                "(version %d <= %d)",
                event.agent_key, event.config_version, current_version,
            )
            return 0

        self._known_versions[event.agent_key] = event.config_version

        payload = json.dumps({
            "agent_key": event.agent_key,
            "old_config": event.old_config,
            "new_config": event.new_config,
            "changed_keys": list(event.changed_keys),
            "config_version": event.config_version,
            "changed_by": event.changed_by,
            "timestamp": event.timestamp.isoformat(),
        }, default=str)

        receivers = await self._redis.publish(self.channel, payload)
        logger.info(
            "ConfigHotReload: published change for '%s' v%d to %d receivers",
            event.agent_key, event.config_version, receivers,
        )
        return receivers

    # ------------------------------------------------------------------
    # Background Listener
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Background task that processes pub/sub messages."""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if message is None:
                    await asyncio.sleep(0.1)
                    continue
                if message["type"] != "message":
                    continue

                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                await self._process_message(data)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "ConfigHotReload: listener error: %s", exc,
                )
                await asyncio.sleep(1.0)

    async def _process_message(self, raw: str) -> None:
        """Parse and dispatch a config change message.

        Args:
            raw: Raw JSON string from Redis.
        """
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("ConfigHotReload: invalid JSON: %s", exc)
            return

        agent_key = payload.get("agent_key", "")
        config_version = payload.get("config_version", 0)

        # Version check: discard stale messages
        current_version = self._known_versions.get(agent_key, 0)
        if config_version <= current_version:
            logger.debug(
                "ConfigHotReload: discarding stale message for '%s' "
                "(v%d <= v%d)",
                agent_key, config_version, current_version,
            )
            return

        self._known_versions[agent_key] = config_version

        event = ConfigChangeEvent(
            agent_key=agent_key,
            old_config=payload.get("old_config", {}),
            new_config=payload.get("new_config", {}),
            changed_keys=frozenset(payload.get("changed_keys", [])),
            config_version=config_version,
            changed_by=payload.get("changed_by", ""),
            timestamp=datetime.fromisoformat(
                payload.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
        )

        logger.info(
            "ConfigHotReload: received change for '%s' v%d (keys: %s)",
            agent_key, config_version, ", ".join(event.changed_keys) or "all",
        )

        # Dispatch to agent-specific callbacks
        agent_callbacks = self._callbacks.get(agent_key, [])
        for cb in agent_callbacks:
            try:
                await cb(event)
            except Exception as exc:
                logger.error(
                    "ConfigHotReload: callback error for '%s': %s",
                    agent_key, exc,
                )

        # Dispatch to global callbacks
        for cb in self._global_callbacks:
            try:
                await cb(event)
            except Exception as exc:
                logger.error(
                    "ConfigHotReload: global callback error: %s", exc,
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot.

        Returns:
            Dictionary with registration and version info.
        """
        return {
            "channel": self.channel,
            "running": self._running,
            "registered_agents": list(self._callbacks.keys()),
            "global_callbacks": len(self._global_callbacks),
            "known_versions": dict(self._known_versions),
        }


__all__ = [
    "ConfigChangeEvent",
    "ConfigHotReload",
    "ReloadCallback",
]
