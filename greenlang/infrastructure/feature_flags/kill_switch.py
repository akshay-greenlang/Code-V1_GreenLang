# -*- coding: utf-8 -*-
"""
Kill Switch - INFRA-008

Provides an emergency kill switch for feature flags using Redis Pub/Sub.
When a flag is killed, the action is broadcast to all application instances
via a Redis channel, and each instance maintains a local in-memory cache of
killed flags for zero-latency checks on the evaluation hot path.

Features:
    - Instant local check: ``is_killed()`` reads from an in-memory dict (no I/O).
    - Redis Pub/Sub: kill/restore events propagate across all pods in < 1 ms.
    - Auto-rollback: optionally schedule automatic restoration after N minutes.
    - JSON event format for debuggability and logging.

Design principles:
    - The ``is_killed()`` check is O(1) with zero network I/O.
    - Redis is optional at startup (graceful degradation if unavailable).
    - Background listener runs as an asyncio task, cleaned up via ``stop_listener()``.

Example:
    >>> ks = KillSwitch(redis_url="redis://localhost:6379/0")
    >>> await ks.start_listener()
    >>> await ks.activate("enable-scope3-calc")
    >>> ks.is_killed("enable-scope3-calc")
    True
    >>> await ks.deactivate("enable-scope3-calc")
    >>> ks.is_killed("enable-scope3-calc")
    False
    >>> await ks.stop_listener()
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class KillSwitch:
    """Emergency kill switch backed by Redis Pub/Sub and a local cache.

    Killed flag keys are stored in a local ``dict`` for zero-latency lookups.
    Kill and restore events are published to a Redis Pub/Sub channel so all
    application instances converge on the same set of killed flags.

    A background asyncio task subscribes to the channel and updates the local
    cache whenever events arrive.

    Attributes:
        _redis_url: Redis connection URL.
        _channel: Redis Pub/Sub channel name.
        _killed_flags: Local cache mapping flag_key -> kill metadata.
        _listener_task: Background asyncio task for Pub/Sub subscription.
        _running: Whether the background listener is active.
        _redis: Lazy-initialized aioredis/redis.asyncio connection.
        _pubsub: Redis Pub/Sub subscription object.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        channel: str = "ff:killswitch",
    ) -> None:
        """Initialize the KillSwitch.

        Args:
            redis_url: Redis connection URL.
            channel: Redis Pub/Sub channel for kill switch events.
        """
        self._redis_url = redis_url
        self._channel = channel
        self._killed_flags: Dict[str, Dict[str, Any]] = {}
        self._listener_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._redis: Any = None
        self._pubsub: Any = None
        self._auto_restore_tasks: Dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def activate(
        self,
        flag_key: str,
        auto_restore_minutes: Optional[int] = None,
    ) -> None:
        """Kill a feature flag (emergency off).

        Immediately updates the local cache and publishes the kill event to
        Redis Pub/Sub so all other instances are notified.

        Args:
            flag_key: The flag key to kill.
            auto_restore_minutes: If set, automatically restore the flag after
                this many minutes. Must be > 0.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        event = {
            "action": "kill",
            "flag_key": flag_key,
            "timestamp": timestamp,
        }
        if auto_restore_minutes is not None and auto_restore_minutes > 0:
            event["auto_restore_minutes"] = auto_restore_minutes

        # Update local cache immediately
        self._killed_flags[flag_key] = {
            "killed_at": timestamp,
            "auto_restore_minutes": auto_restore_minutes,
        }

        logger.info(
            "KillSwitch: activated for flag '%s' (auto_restore=%s min)",
            flag_key,
            auto_restore_minutes,
        )

        # Publish to Redis
        await self._publish(event)

        # Schedule auto-restore if requested
        if auto_restore_minutes is not None and auto_restore_minutes > 0:
            self._schedule_auto_restore(flag_key, auto_restore_minutes)

    async def deactivate(self, flag_key: str) -> None:
        """Restore a killed feature flag.

        Removes the flag from the local killed cache and publishes a restore
        event to Redis Pub/Sub.

        Args:
            flag_key: The flag key to restore.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        event = {
            "action": "restore",
            "flag_key": flag_key,
            "timestamp": timestamp,
        }

        # Update local cache immediately
        self._killed_flags.pop(flag_key, None)

        # Cancel any pending auto-restore
        self._cancel_auto_restore(flag_key)

        logger.info("KillSwitch: deactivated for flag '%s'", flag_key)

        # Publish to Redis
        await self._publish(event)

    def is_killed(self, flag_key: str) -> bool:
        """Check if a flag is currently killed.

        This is a pure in-memory lookup with O(1) complexity and zero
        network I/O. It is safe to call on every flag evaluation.

        Args:
            flag_key: The flag key to check.

        Returns:
            True if the flag is killed, False otherwise.
        """
        return flag_key in self._killed_flags

    def get_killed_flags(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of all currently killed flags with metadata.

        Returns:
            Dict mapping flag_key to kill metadata (killed_at, auto_restore_minutes).
        """
        return dict(self._killed_flags)

    # ------------------------------------------------------------------
    # Background listener
    # ------------------------------------------------------------------

    async def start_listener(self) -> None:
        """Start the background Redis Pub/Sub listener.

        Subscribes to the kill switch channel and processes incoming events
        in an asyncio background task. If Redis is unavailable, logs a
        warning and returns without starting the listener (graceful
        degradation).
        """
        if self._running:
            logger.debug("KillSwitch: listener is already running")
            return

        try:
            redis_module = self._get_redis_module()
            if redis_module is None:
                logger.warning(
                    "KillSwitch: redis.asyncio not available. "
                    "Kill switch will operate in local-only mode."
                )
                return

            self._redis = redis_module.from_url(
                self._redis_url,
                decode_responses=True,
            )
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(self._channel)

            self._running = True
            self._listener_task = asyncio.create_task(
                self._listen_loop(),
                name=f"killswitch-listener-{self._channel}",
            )
            logger.info(
                "KillSwitch: listener started on channel '%s'",
                self._channel,
            )

        except Exception as exc:
            logger.warning(
                "KillSwitch: failed to start listener: %s. "
                "Operating in local-only mode.",
                exc,
            )
            self._running = False

    async def stop_listener(self) -> None:
        """Stop the background Redis Pub/Sub listener and release resources."""
        self._running = False

        # Cancel all auto-restore tasks
        for flag_key in list(self._auto_restore_tasks):
            self._cancel_auto_restore(flag_key)

        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe(self._channel)
                await self._pubsub.close()
            except Exception as exc:
                logger.debug("KillSwitch: error closing pubsub: %s", exc)
            self._pubsub = None

        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception as exc:
                logger.debug("KillSwitch: error closing redis: %s", exc)
            self._redis = None

        logger.info("KillSwitch: listener stopped")

    # ------------------------------------------------------------------
    # Internal: Pub/Sub loop
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Background loop that reads messages from the Redis Pub/Sub channel.

        Runs until ``_running`` is set to False or the task is cancelled.
        Each message is parsed as JSON and dispatched to the local cache
        update handler.
        """
        logger.debug("KillSwitch: entering listen loop")
        try:
            while self._running:
                try:
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if message is not None and message.get("type") == "message":
                        self._handle_message(message["data"])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "KillSwitch: error reading pubsub message: %s", exc
                    )
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.debug("KillSwitch: listen loop cancelled")

    def _handle_message(self, raw_data: str) -> None:
        """Handle an incoming Pub/Sub message by updating the local cache.

        Expected JSON format::

            {"action": "kill"|"restore", "flag_key": "...", "timestamp": "..."}

        Args:
            raw_data: Raw JSON string from the Pub/Sub channel.
        """
        try:
            event: Dict[str, Any] = json.loads(raw_data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "KillSwitch: failed to parse message: %s (raw=%s)",
                exc,
                raw_data,
            )
            return

        action = event.get("action")
        flag_key = event.get("flag_key")

        if not action or not flag_key:
            logger.debug(
                "KillSwitch: ignoring malformed event: %s", event
            )
            return

        if action == "kill":
            auto_restore = event.get("auto_restore_minutes")
            self._killed_flags[flag_key] = {
                "killed_at": event.get("timestamp", ""),
                "auto_restore_minutes": auto_restore,
            }
            logger.info(
                "KillSwitch: received kill event for flag '%s'", flag_key
            )
            # Schedule auto-restore if specified
            if auto_restore is not None and auto_restore > 0:
                self._schedule_auto_restore(flag_key, auto_restore)

        elif action == "restore":
            self._killed_flags.pop(flag_key, None)
            self._cancel_auto_restore(flag_key)
            logger.info(
                "KillSwitch: received restore event for flag '%s'", flag_key
            )

        else:
            logger.debug(
                "KillSwitch: ignoring unknown action '%s' for flag '%s'",
                action,
                flag_key,
            )

    # ------------------------------------------------------------------
    # Internal: Publish
    # ------------------------------------------------------------------

    async def _publish(self, event: Dict[str, Any]) -> None:
        """Publish an event to the Redis Pub/Sub channel.

        If Redis is not connected, the event is logged but not published
        (local-only mode).

        Args:
            event: The event dict to serialize as JSON and publish.
        """
        if self._redis is None:
            logger.debug(
                "KillSwitch: Redis not connected, event not published: %s",
                event,
            )
            return

        try:
            payload = json.dumps(event)
            await self._redis.publish(self._channel, payload)
            logger.debug(
                "KillSwitch: published event to '%s': %s",
                self._channel,
                payload,
            )
        except Exception as exc:
            logger.warning(
                "KillSwitch: failed to publish event: %s", exc
            )

    # ------------------------------------------------------------------
    # Internal: Auto-restore
    # ------------------------------------------------------------------

    def _schedule_auto_restore(self, flag_key: str, minutes: int) -> None:
        """Schedule an automatic restore after the specified number of minutes.

        Cancels any existing auto-restore task for the same flag.

        Args:
            flag_key: The flag key to auto-restore.
            minutes: Number of minutes until restoration.
        """
        self._cancel_auto_restore(flag_key)

        async def _auto_restore() -> None:
            await asyncio.sleep(minutes * 60)
            logger.info(
                "KillSwitch: auto-restoring flag '%s' after %d minutes",
                flag_key,
                minutes,
            )
            await self.deactivate(flag_key)

        task = asyncio.create_task(
            _auto_restore(),
            name=f"killswitch-auto-restore-{flag_key}",
        )
        self._auto_restore_tasks[flag_key] = task

    def _cancel_auto_restore(self, flag_key: str) -> None:
        """Cancel a pending auto-restore task if one exists.

        Args:
            flag_key: The flag key whose auto-restore to cancel.
        """
        task = self._auto_restore_tasks.pop(flag_key, None)
        if task is not None and not task.done():
            task.cancel()
            logger.debug(
                "KillSwitch: cancelled auto-restore for flag '%s'", flag_key
            )

    # ------------------------------------------------------------------
    # Internal: Redis module loading
    # ------------------------------------------------------------------

    @staticmethod
    def _get_redis_module() -> Any:
        """Attempt to import redis.asyncio.

        Returns:
            The redis.asyncio module, or None if not installed.
        """
        try:
            import redis.asyncio as aioredis
            return aioredis
        except ImportError:
            return None
