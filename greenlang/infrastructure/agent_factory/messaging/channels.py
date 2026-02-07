# -*- coding: utf-8 -*-
"""
Channel Manager - Named channel registry with topic-based routing.

Manages named communication channels for the inter-agent messaging system.
Each channel has a type (durable or ephemeral), a configuration, and runtime
metadata (subscriber count, message count).  Channel definitions are persisted
in Redis hashes so that all nodes in the cluster share a consistent view.

The ChannelManager provides CRUD operations, channel discovery, and automatic
resource cleanup for deleted channels.

Classes:
    - ChannelConfig: Per-channel configuration.
    - Channel: Runtime channel descriptor.
    - ChannelManager: Named channel registry.

Example:
    >>> manager = ChannelManager(redis_client)
    >>> ch = await manager.create_channel("carbon.calculate", ChannelType.DURABLE)
    >>> channels = await manager.list_channels()
    >>> await manager.delete_channel("carbon.calculate")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.agent_factory.messaging.protocol import (
    ChannelType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    """Per-channel configuration.

    Controls message size limits, TTL, retry behavior, and serialization
    format.

    Attributes:
        max_message_size_bytes: Maximum payload size in bytes (default 1 MB).
        message_ttl_seconds: Time-to-live for durable messages (default 30m).
        max_retries: Maximum delivery retries for durable channels.
        max_subscribers: Maximum concurrent subscribers for ephemeral channels.
        serialization: Serialization format (``"json"`` or ``"msgpack"``).
        max_stream_length: Approximate max entries in the backing Redis Stream
            (durable only).
        consumer_group_prefix: Override for consumer group naming.
        description: Human-readable description of the channel.
    """

    max_message_size_bytes: int = 1_048_576  # 1 MB
    message_ttl_seconds: int = 1800  # 30 minutes
    max_retries: int = 3
    max_subscribers: int = 1000
    serialization: str = "json"
    max_stream_length: int = 100_000
    consumer_group_prefix: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "max_message_size_bytes": self.max_message_size_bytes,
            "message_ttl_seconds": self.message_ttl_seconds,
            "max_retries": self.max_retries,
            "max_subscribers": self.max_subscribers,
            "serialization": self.serialization,
            "max_stream_length": self.max_stream_length,
            "consumer_group_prefix": self.consumer_group_prefix,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChannelConfig:
        """Deserialize configuration from a plain dictionary.

        Args:
            data: Dictionary with config fields.

        Returns:
            ChannelConfig instance.
        """
        return cls(
            max_message_size_bytes=int(
                data.get("max_message_size_bytes", 1_048_576)
            ),
            message_ttl_seconds=int(data.get("message_ttl_seconds", 1800)),
            max_retries=int(data.get("max_retries", 3)),
            max_subscribers=int(data.get("max_subscribers", 1000)),
            serialization=data.get("serialization", "json"),
            max_stream_length=int(data.get("max_stream_length", 100_000)),
            consumer_group_prefix=data.get("consumer_group_prefix", ""),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Channel Descriptor
# ---------------------------------------------------------------------------


@dataclass
class Channel:
    """Runtime descriptor for a named channel.

    Combines the channel's identity, type, configuration, and runtime
    statistics into a single object.

    Attributes:
        name: Unique channel name (e.g. ``"carbon.calculate"``).
        channel_type: Transport type (durable or ephemeral).
        config: Channel configuration.
        created_at: UTC timestamp when the channel was registered.
        updated_at: UTC timestamp of the last configuration update.
        subscriber_count: Current number of subscribers (ephemeral only).
        message_count: Approximate number of messages in the backing store.
        status: Channel status (``"active"`` or ``"inactive"``).
    """

    name: str
    channel_type: ChannelType
    config: ChannelConfig = field(default_factory=ChannelConfig)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    subscriber_count: int = 0
    message_count: int = 0
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the channel to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON or Redis storage.
        """
        return {
            "name": self.name,
            "channel_type": self.channel_type.value,
            "config": json.dumps(self.config.to_dict()),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "subscriber_count": str(self.subscriber_count),
            "message_count": str(self.message_count),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Channel:
        """Deserialize a channel from a plain dictionary.

        Args:
            data: Dictionary with channel fields.

        Returns:
            Channel instance.
        """
        # Parse config (may be a JSON string or a dict)
        config_raw = data.get("config", "{}")
        if isinstance(config_raw, str):
            config_data = json.loads(config_raw)
        else:
            config_data = config_raw

        # Parse timestamps
        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(timezone.utc)

        updated_at = data.get("updated_at", "")
        if isinstance(updated_at, str) and updated_at:
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now(timezone.utc)

        return cls(
            name=data.get("name", ""),
            channel_type=ChannelType(
                data.get("channel_type", ChannelType.EPHEMERAL.value)
            ),
            config=ChannelConfig.from_dict(config_data),
            created_at=created_at,
            updated_at=updated_at,
            subscriber_count=int(data.get("subscriber_count", 0)),
            message_count=int(data.get("message_count", 0)),
            status=data.get("status", "active"),
        )

    @property
    def is_durable(self) -> bool:
        """Check whether this is a durable channel.

        Returns:
            True if the channel uses Redis Streams.
        """
        return self.channel_type == ChannelType.DURABLE

    @property
    def is_ephemeral(self) -> bool:
        """Check whether this is an ephemeral channel.

        Returns:
            True if the channel uses Redis Pub/Sub.
        """
        return self.channel_type == ChannelType.EPHEMERAL

    @property
    def consumer_group(self) -> str:
        """Get the consumer group name for this durable channel.

        Returns:
            Consumer group name string.
        """
        prefix = self.config.consumer_group_prefix
        if prefix:
            return f"{prefix}-{self.name}-consumers"
        return f"{self.name}-consumers"


# ---------------------------------------------------------------------------
# ChannelManager
# ---------------------------------------------------------------------------


# Redis key constants
_REGISTRY_KEY = "gl:channels:registry"
_CHANNEL_PREFIX = "gl:channels:meta:"


class ChannelManager:
    """Named channel registry with topic-based routing.

    Manages the lifecycle of named channels across the GreenLang cluster.
    Channel metadata is stored in Redis hashes so that all nodes share a
    consistent view.  The manager supports:

    - **create_channel**: Register a new channel with configuration.
    - **get_channel**: Look up a channel by name.
    - **update_channel**: Modify a channel's configuration.
    - **list_channels**: List all registered channels with optional filtering.
    - **delete_channel**: Remove a channel and clean up its resources.
    - **channel_exists**: Check whether a channel is registered.
    - **update_stats**: Refresh subscriber and message counts.

    All Redis operations are performed through the injected client.

    Attributes:
        stream_prefix: Prefix for durable stream keys.
        pubsub_prefix: Prefix for ephemeral channel keys.
    """

    def __init__(
        self,
        redis_client: Any,
        stream_prefix: str = "gl:msg:",
        pubsub_prefix: str = "gl:evt:",
    ) -> None:
        """Initialize the channel manager.

        Args:
            redis_client: Async Redis client (dependency injection).
            stream_prefix: Prefix for durable Redis Stream keys.
            pubsub_prefix: Prefix for ephemeral Pub/Sub channel keys.
        """
        self._redis = redis_client
        self.stream_prefix = stream_prefix
        self.pubsub_prefix = pubsub_prefix
        logger.debug(
            "ChannelManager initialized (stream_prefix=%s, pubsub_prefix=%s)",
            stream_prefix,
            pubsub_prefix,
        )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_channel(
        self,
        name: str,
        channel_type: ChannelType,
        config: Optional[ChannelConfig] = None,
    ) -> Channel:
        """Create or update a named channel.

        If a channel with the given name already exists, its configuration
        is updated and the existing channel is returned.

        Args:
            name: Unique channel name (e.g. ``"carbon.calculate"``).
            channel_type: Transport type for this channel.
            config: Channel configuration. Defaults are used if None.

        Returns:
            The created or updated Channel.
        """
        existing = await self.get_channel(name)
        if existing is not None:
            logger.info(
                "Channel '%s' already exists; updating configuration", name
            )
            existing.config = config or existing.config
            existing.updated_at = datetime.now(timezone.utc)
            await self._persist_channel(existing)
            return existing

        channel = Channel(
            name=name,
            channel_type=channel_type,
            config=config or ChannelConfig(),
        )

        # Set up infrastructure for durable channels
        if channel_type == ChannelType.DURABLE:
            await self._setup_durable_infrastructure(channel)

        await self._persist_channel(channel)

        logger.info(
            "Channel '%s' created (type=%s)", name, channel_type.value
        )
        return channel

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_channel(self, name: str) -> Optional[Channel]:
        """Get a channel by name.

        Args:
            name: Channel name.

        Returns:
            Channel instance or None if not found.
        """
        meta_key = f"{_CHANNEL_PREFIX}{name}"
        data = await self._redis.hgetall(meta_key)
        if not data:
            return None
        return Channel.from_dict(data)

    async def channel_exists(self, name: str) -> bool:
        """Check whether a channel is registered.

        Args:
            name: Channel name.

        Returns:
            True if the channel exists.
        """
        meta_key = f"{_CHANNEL_PREFIX}{name}"
        return bool(await self._redis.exists(meta_key))

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    async def list_channels(
        self,
        channel_type: Optional[ChannelType] = None,
        status: Optional[str] = None,
    ) -> List[Channel]:
        """List all registered channels with optional filtering.

        Args:
            channel_type: Filter by transport type.
            status: Filter by status (``"active"`` or ``"inactive"``).

        Returns:
            List of Channel instances.
        """
        # Get all registered channel names from the registry set
        names = await self._redis.smembers(_REGISTRY_KEY)
        if not names:
            return []

        channels: List[Channel] = []
        for name in sorted(names):
            channel = await self.get_channel(name)
            if channel is None:
                continue
            if channel_type is not None and channel.channel_type != channel_type:
                continue
            if status is not None and channel.status != status:
                continue
            channels.append(channel)

        return channels

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_channel(
        self,
        name: str,
        config: ChannelConfig,
    ) -> Optional[Channel]:
        """Update the configuration of an existing channel.

        Args:
            name: Channel name.
            config: New channel configuration.

        Returns:
            Updated Channel or None if the channel does not exist.
        """
        channel = await self.get_channel(name)
        if channel is None:
            logger.warning(
                "Cannot update channel '%s': not found", name
            )
            return None

        channel.config = config
        channel.updated_at = datetime.now(timezone.utc)
        await self._persist_channel(channel)

        logger.info("Channel '%s' configuration updated", name)
        return channel

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_channel(self, name: str) -> bool:
        """Delete a channel and clean up its Redis resources.

        For durable channels, the backing Redis Stream and consumer groups
        are deleted.  For ephemeral channels, only the metadata is removed.

        Args:
            name: Channel name.

        Returns:
            True if the channel was found and deleted.
        """
        channel = await self.get_channel(name)
        if channel is None:
            logger.warning(
                "Cannot delete channel '%s': not found", name
            )
            return False

        # Clean up transport resources
        if channel.is_durable:
            await self._cleanup_durable_infrastructure(channel)

        # Remove metadata
        meta_key = f"{_CHANNEL_PREFIX}{name}"
        await self._redis.delete(meta_key)
        await self._redis.srem(_REGISTRY_KEY, name)

        logger.info("Channel '%s' deleted", name)
        return True

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def update_stats(self, name: str) -> Optional[Channel]:
        """Refresh the subscriber and message counts for a channel.

        Queries the backing Redis data structure for current statistics.

        Args:
            name: Channel name.

        Returns:
            Updated Channel or None if the channel does not exist.
        """
        channel = await self.get_channel(name)
        if channel is None:
            return None

        if channel.is_durable:
            stream_key = f"{self.stream_prefix}{name}"
            try:
                channel.message_count = await self._redis.xlen(stream_key)
            except Exception:
                channel.message_count = 0

            # Subscriber count for durable = number of consumers in the group
            try:
                info = await self._redis.xinfo_groups(stream_key)
                if info:
                    channel.subscriber_count = sum(
                        g.get("consumers", 0) for g in info
                    )
            except Exception:
                channel.subscriber_count = 0

        elif channel.is_ephemeral:
            # Pub/Sub subscriber count via PUBSUB NUMSUB
            pubsub_key = f"{self.pubsub_prefix}{name}"
            try:
                numsub = await self._redis.pubsub_numsub(pubsub_key)
                # numsub returns [(channel, count), ...]
                if numsub:
                    channel.subscriber_count = numsub[0][1] if isinstance(
                        numsub[0], (list, tuple)
                    ) else 0
            except Exception:
                channel.subscriber_count = 0

        channel.updated_at = datetime.now(timezone.utc)
        await self._persist_channel(channel)
        return channel

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    async def create_channels_bulk(
        self,
        definitions: List[Dict[str, Any]],
    ) -> List[Channel]:
        """Create multiple channels from a list of definitions.

        Each definition is a dictionary with ``name``, ``channel_type``
        (string), and optional ``config`` (dictionary).

        Args:
            definitions: List of channel definition dictionaries.

        Returns:
            List of created Channel instances.
        """
        channels: List[Channel] = []
        for defn in definitions:
            name = defn.get("name", "")
            if not name:
                logger.warning("Skipping channel definition with empty name")
                continue
            channel_type = ChannelType(
                defn.get("channel_type", ChannelType.EPHEMERAL.value)
            )
            config_data = defn.get("config")
            config = (
                ChannelConfig.from_dict(config_data)
                if config_data
                else None
            )
            channel = await self.create_channel(name, channel_type, config)
            channels.append(channel)
        return channels

    async def delete_all_channels(self) -> int:
        """Delete all registered channels. Primarily for testing.

        Returns:
            Number of channels deleted.
        """
        channels = await self.list_channels()
        count = 0
        for channel in channels:
            deleted = await self.delete_channel(channel.name)
            if deleted:
                count += 1
        logger.info("Deleted %d channels", count)
        return count

    # ------------------------------------------------------------------
    # Predefined Channel Templates
    # ------------------------------------------------------------------

    async def create_agent_channels(self, agent_key: str) -> Dict[str, Channel]:
        """Create the standard set of channels for an agent.

        Each agent gets:
        - ``{agent_key}.inbox``: Durable inbox for receiving work.
        - ``{agent_key}.outbox``: Durable outbox for sending results.
        - ``{agent_key}.events``: Ephemeral channel for telemetry/health.

        Args:
            agent_key: Agent identifier.

        Returns:
            Dictionary mapping channel purpose to Channel.
        """
        inbox = await self.create_channel(
            f"{agent_key}.inbox",
            ChannelType.DURABLE,
            ChannelConfig(
                description=f"Inbox for {agent_key}",
                message_ttl_seconds=1800,
            ),
        )
        outbox = await self.create_channel(
            f"{agent_key}.outbox",
            ChannelType.DURABLE,
            ChannelConfig(
                description=f"Outbox for {agent_key}",
                message_ttl_seconds=1800,
            ),
        )
        events = await self.create_channel(
            f"{agent_key}.events",
            ChannelType.EPHEMERAL,
            ChannelConfig(
                description=f"Events for {agent_key}",
                max_subscribers=500,
            ),
        )

        return {
            "inbox": inbox,
            "outbox": outbox,
            "events": events,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _persist_channel(self, channel: Channel) -> None:
        """Write channel metadata to Redis.

        Args:
            channel: Channel to persist.
        """
        meta_key = f"{_CHANNEL_PREFIX}{channel.name}"
        await self._redis.hset(meta_key, mapping=channel.to_dict())
        await self._redis.sadd(_REGISTRY_KEY, channel.name)

    async def _setup_durable_infrastructure(self, channel: Channel) -> None:
        """Create the backing Redis Stream and consumer group for a durable channel.

        Args:
            channel: Durable channel descriptor.
        """
        stream_key = f"{self.stream_prefix}{channel.name}"
        group = channel.consumer_group
        try:
            await self._redis.xgroup_create(
                stream_key,
                group,
                id="0",
                mkstream=True,
            )
            logger.debug(
                "Created stream %s with consumer group %s",
                stream_key,
                group,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.debug(
                "Stream %s / group %s already exists",
                stream_key,
                group,
            )

    async def _cleanup_durable_infrastructure(self, channel: Channel) -> None:
        """Delete the backing Redis Stream for a durable channel.

        Args:
            channel: Durable channel descriptor.
        """
        stream_key = f"{self.stream_prefix}{channel.name}"
        try:
            await self._redis.delete(stream_key)
            logger.debug("Deleted stream %s", stream_key)
        except Exception as exc:
            logger.warning(
                "Failed to delete stream %s: %s", stream_key, exc
            )

        # Also clean up the DLQ stream
        dlq_key = f"gl:dlq:{stream_key}"
        try:
            await self._redis.delete(dlq_key)
        except Exception:
            pass


__all__ = [
    "Channel",
    "ChannelConfig",
    "ChannelManager",
]
