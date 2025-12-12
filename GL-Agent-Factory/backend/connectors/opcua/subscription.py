"""
OPC-UA Subscription Handler for GreenLang Process Heat Agents.

This module provides real-time data subscription functionality for OPC-UA servers,
enabling continuous monitoring of process variables, temperatures, pressures,
and other sensor data from industrial systems (DCS, PLC, SCADA).

Features:
- Async subscription management
- Configurable sampling and publishing intervals
- Data change callbacks with quality handling
- Automatic reconnection and recovery
- Deadband filtering (absolute and percent)
- Queue management for high-frequency data

Usage:
    from connectors.opcua.subscription import OPCUASubscription, SubscriptionManager

    # Create subscription
    sub = OPCUASubscription(
        subscription_id="furnace_temps",
        publishing_interval_ms=1000,
    )

    # Add monitored items
    await sub.add_monitored_items([
        "ns=2;s=Furnace1.Temperature.PV",
        "ns=2;s=Furnace1.Pressure.PV",
    ])

    # Set callback for data changes
    sub.on_data_change(handle_temperature_change)

    # Start subscription
    await sub.start()
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
import uuid
from collections import deque

from pydantic import BaseModel, Field

from .types import (
    NodeValue,
    DataProvenance,
    OPCUAQuality,
    QualityLevel,
    MonitoredItemConfig,
    SubscriptionConfig,
    DataChangeNotification,
    MonitoringMode,
    DataChangeTrigger,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Callback Types
# =============================================================================

# Type alias for data change callbacks
DataChangeCallback = Callable[[DataChangeNotification], Coroutine[Any, Any, None]]
StatusChangeCallback = Callable[[str, str], Coroutine[Any, Any, None]]


# =============================================================================
# Subscription Statistics
# =============================================================================


@dataclass
class SubscriptionStatistics:
    """Statistics for a subscription."""

    subscription_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_publish_time: Optional[datetime] = None
    publish_count: int = 0
    notification_count: int = 0
    data_change_count: int = 0
    keep_alive_count: int = 0
    late_publish_count: int = 0
    monitored_item_count: int = 0
    queue_overflow_count: int = 0
    republish_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "subscription_id": self.subscription_id,
            "created_at": self.created_at.isoformat(),
            "last_publish_time": self.last_publish_time.isoformat() if self.last_publish_time else None,
            "publish_count": self.publish_count,
            "notification_count": self.notification_count,
            "data_change_count": self.data_change_count,
            "keep_alive_count": self.keep_alive_count,
            "late_publish_count": self.late_publish_count,
            "monitored_item_count": self.monitored_item_count,
            "queue_overflow_count": self.queue_overflow_count,
            "republish_count": self.republish_count,
        }


# =============================================================================
# Monitored Item
# =============================================================================


class MonitoredItem:
    """
    Represents a monitored item in an OPC-UA subscription.

    Handles sampling, deadband filtering, and queue management for a single node.
    """

    def __init__(
        self,
        config: MonitoredItemConfig,
        client_handle: int,
    ):
        """
        Initialize monitored item.

        Args:
            config: Monitoring configuration
            client_handle: Client-side handle for this item
        """
        self.config = config
        self.client_handle = client_handle
        self.server_handle: Optional[int] = None

        # Value queue
        self._queue: deque = deque(maxlen=config.queue_size)
        self._last_value: Optional[NodeValue] = None
        self._last_reported_value: Optional[NodeValue] = None

        # Statistics
        self.sample_count: int = 0
        self.report_count: int = 0
        self.filtered_count: int = 0
        self.overflow_count: int = 0

        # State
        self._enabled: bool = True
        self._sampling: bool = False

    @property
    def node_id(self) -> str:
        """Get the node ID being monitored."""
        return self.config.node_id

    @property
    def last_value(self) -> Optional[NodeValue]:
        """Get the last sampled value."""
        return self._last_value

    def apply_value(self, value: NodeValue) -> bool:
        """
        Apply a new value and determine if it should be reported.

        Args:
            value: New value from OPC-UA server

        Returns:
            True if value should be reported (passed deadband filter)
        """
        self.sample_count += 1
        self._last_value = value

        # Check if monitoring is disabled
        if self.config.monitoring_mode == MonitoringMode.DISABLED:
            return False

        # Check if value passes deadband filter
        if not self._passes_deadband(value):
            self.filtered_count += 1
            return False

        # Check data change trigger
        if not self._triggers_change(value):
            return False

        # Add to queue
        if len(self._queue) >= self.config.queue_size:
            if self.config.discard_oldest:
                self._queue.popleft()
            else:
                self.overflow_count += 1
                return False

        self._queue.append(value)
        self._last_reported_value = value
        self.report_count += 1

        return True

    def _passes_deadband(self, value: NodeValue) -> bool:
        """
        Check if value passes deadband filter.

        Args:
            value: New value to check

        Returns:
            True if value passes deadband filter
        """
        if not self.config.deadband_type or self.config.deadband_value <= 0:
            return True

        if self._last_reported_value is None:
            return True

        try:
            new_val = float(value.value)
            old_val = float(self._last_reported_value.value)
        except (TypeError, ValueError):
            # Non-numeric values always pass
            return True

        if self.config.deadband_type == "Absolute":
            return abs(new_val - old_val) > self.config.deadband_value

        elif self.config.deadband_type == "Percent":
            if old_val == 0:
                return new_val != 0
            percent_change = abs((new_val - old_val) / old_val) * 100
            return percent_change > self.config.deadband_value

        return True

    def _triggers_change(self, value: NodeValue) -> bool:
        """
        Check if value triggers a data change based on trigger condition.

        Args:
            value: New value to check

        Returns:
            True if data change should be triggered
        """
        if self._last_reported_value is None:
            return True

        trigger = self.config.data_change_trigger

        if trigger == DataChangeTrigger.STATUS:
            return value.quality != self._last_reported_value.quality

        elif trigger == DataChangeTrigger.STATUS_VALUE:
            return (
                value.quality != self._last_reported_value.quality or
                value.value != self._last_reported_value.value
            )

        elif trigger == DataChangeTrigger.STATUS_VALUE_TIMESTAMP:
            return (
                value.quality != self._last_reported_value.quality or
                value.value != self._last_reported_value.value or
                value.source_timestamp != self._last_reported_value.source_timestamp
            )

        return True

    def get_pending_values(self) -> List[NodeValue]:
        """
        Get all pending values from the queue.

        Returns:
            List of pending values
        """
        values = list(self._queue)
        self._queue.clear()
        return values

    def enable(self) -> None:
        """Enable monitoring."""
        self._enabled = True
        logger.debug(f"Enabled monitoring for {self.node_id}")

    def disable(self) -> None:
        """Disable monitoring."""
        self._enabled = False
        logger.debug(f"Disabled monitoring for {self.node_id}")


# =============================================================================
# OPC-UA Subscription
# =============================================================================


class OPCUASubscription:
    """
    OPC-UA Subscription for real-time data monitoring.

    Manages a subscription to an OPC-UA server with configurable
    publishing intervals, monitored items, and data change callbacks.
    """

    def __init__(
        self,
        config: Optional[SubscriptionConfig] = None,
        subscription_id: Optional[str] = None,
        publishing_interval_ms: int = 1000,
        endpoint_url: str = "",
    ):
        """
        Initialize OPC-UA subscription.

        Args:
            config: Full subscription configuration
            subscription_id: Unique subscription identifier
            publishing_interval_ms: Publishing interval in milliseconds
            endpoint_url: OPC-UA server endpoint URL
        """
        if config:
            self.config = config
        else:
            self.config = SubscriptionConfig(
                subscription_id=subscription_id or str(uuid.uuid4()),
                publishing_interval_ms=publishing_interval_ms,
            )

        self.endpoint_url = endpoint_url

        # Monitored items
        self._monitored_items: Dict[str, MonitoredItem] = {}
        self._client_handle_counter: int = 0

        # Callbacks
        self._data_change_callbacks: List[DataChangeCallback] = []
        self._status_change_callbacks: List[StatusChangeCallback] = []

        # State
        self._active: bool = False
        self._publishing_enabled: bool = True
        self._sequence_number: int = 0

        # Statistics
        self.statistics = SubscriptionStatistics(
            subscription_id=self.config.subscription_id
        )

        # Background task
        self._publish_task: Optional[asyncio.Task] = None

        # OPC-UA client reference (set when attached to client)
        self._opcua_subscription: Any = None

    @property
    def subscription_id(self) -> str:
        """Get the subscription ID."""
        return self.config.subscription_id

    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self._active

    @property
    def monitored_item_count(self) -> int:
        """Get number of monitored items."""
        return len(self._monitored_items)

    async def add_monitored_items(
        self,
        nodes: List[Union[str, MonitoredItemConfig]],
        sampling_interval_ms: int = 1000,
    ) -> List[MonitoredItem]:
        """
        Add nodes to be monitored.

        Args:
            nodes: Node IDs or monitoring configurations
            sampling_interval_ms: Default sampling interval

        Returns:
            List of created monitored items
        """
        created_items = []

        for node in nodes:
            if isinstance(node, str):
                # Create default configuration
                config = MonitoredItemConfig(
                    node_id=node,
                    sampling_interval_ms=sampling_interval_ms,
                )
            else:
                config = node

            # Check if already monitoring
            if config.node_id in self._monitored_items:
                logger.warning(f"Node {config.node_id} is already being monitored")
                continue

            # Create monitored item
            self._client_handle_counter += 1
            item = MonitoredItem(config, self._client_handle_counter)

            self._monitored_items[config.node_id] = item
            created_items.append(item)

            logger.info(
                f"Added monitored item: {config.node_id} "
                f"(sampling={config.sampling_interval_ms}ms)"
            )

        self.statistics.monitored_item_count = len(self._monitored_items)

        # If subscription is active and connected, add to server
        if self._active and self._opcua_subscription:
            await self._add_items_to_server(created_items)

        return created_items

    async def remove_monitored_items(self, node_ids: List[str]) -> None:
        """
        Remove nodes from monitoring.

        Args:
            node_ids: Node IDs to stop monitoring
        """
        for node_id in node_ids:
            if node_id in self._monitored_items:
                item = self._monitored_items.pop(node_id)
                logger.info(f"Removed monitored item: {node_id}")

                # Remove from server if active
                if self._active and self._opcua_subscription and item.server_handle:
                    await self._remove_items_from_server([item])

        self.statistics.monitored_item_count = len(self._monitored_items)

    async def set_publishing_interval(self, interval_ms: int) -> None:
        """
        Set the publishing interval.

        Args:
            interval_ms: Publishing interval in milliseconds
        """
        old_interval = self.config.publishing_interval_ms
        self.config.publishing_interval_ms = interval_ms

        logger.info(
            f"Publishing interval changed: {old_interval}ms -> {interval_ms}ms"
        )

        # Update on server if active
        if self._active and self._opcua_subscription:
            await self._update_subscription_on_server()

    def on_data_change(self, callback: DataChangeCallback) -> None:
        """
        Register a callback for data change notifications.

        Args:
            callback: Async callback function
        """
        self._data_change_callbacks.append(callback)
        logger.debug(f"Registered data change callback: {callback.__name__}")

    def on_status_change(self, callback: StatusChangeCallback) -> None:
        """
        Register a callback for subscription status changes.

        Args:
            callback: Async callback function
        """
        self._status_change_callbacks.append(callback)
        logger.debug(f"Registered status change callback: {callback.__name__}")

    def remove_callback(self, callback: Union[DataChangeCallback, StatusChangeCallback]) -> bool:
        """
        Remove a registered callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if callback in self._data_change_callbacks:
            self._data_change_callbacks.remove(callback)
            return True
        if callback in self._status_change_callbacks:
            self._status_change_callbacks.remove(callback)
            return True
        return False

    async def start(self) -> None:
        """Start the subscription."""
        if self._active:
            logger.warning("Subscription is already active")
            return

        self._active = True
        self._publishing_enabled = True

        logger.info(
            f"Starting subscription {self.subscription_id} "
            f"(publishing_interval={self.config.publishing_interval_ms}ms, "
            f"items={len(self._monitored_items)})"
        )

        # Notify status change
        await self._notify_status_change("started", "Subscription started")

    async def stop(self) -> None:
        """Stop the subscription."""
        if not self._active:
            return

        self._active = False
        self._publishing_enabled = False

        # Cancel publish task
        if self._publish_task:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass
            self._publish_task = None

        logger.info(f"Stopped subscription {self.subscription_id}")

        # Notify status change
        await self._notify_status_change("stopped", "Subscription stopped")

    async def enable_publishing(self) -> None:
        """Enable publishing for this subscription."""
        self._publishing_enabled = True
        logger.info(f"Publishing enabled for subscription {self.subscription_id}")

    async def disable_publishing(self) -> None:
        """Disable publishing for this subscription."""
        self._publishing_enabled = False
        logger.info(f"Publishing disabled for subscription {self.subscription_id}")

    async def process_data_change(
        self,
        node_id: str,
        value: Any,
        quality: OPCUAQuality = OPCUAQuality.GOOD,
        source_timestamp: Optional[datetime] = None,
        server_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Process a data change from the OPC-UA server.

        This method is called by the OPC-UA client when data changes.

        Args:
            node_id: Node ID that changed
            value: New value
            quality: OPC-UA quality code
            source_timestamp: Source timestamp
            server_timestamp: Server timestamp
        """
        if node_id not in self._monitored_items:
            logger.warning(f"Data change for unknown node: {node_id}")
            return

        item = self._monitored_items[node_id]

        # Create node value with provenance
        node_value = NodeValue(
            node_id=node_id,
            value=value,
            quality=quality,
            quality_level=QualityLevel.from_quality_code(quality),
            source_timestamp=source_timestamp or datetime.utcnow(),
            server_timestamp=server_timestamp or datetime.utcnow(),
            provenance=DataProvenance(
                source_endpoint=self.endpoint_url,
                source_node_id=node_id,
                retrieval_method="subscription",
                session_id=self.subscription_id,
            ),
        )

        # Apply to monitored item
        if item.apply_value(node_value):
            self.statistics.data_change_count += 1

            # If publishing is enabled, notify immediately for reporting mode
            if self._publishing_enabled and item.config.monitoring_mode == MonitoringMode.REPORTING:
                await self._publish_notification([node_value])

    async def _publish_notification(self, values: List[NodeValue]) -> None:
        """
        Publish a data change notification to callbacks.

        Args:
            values: Changed values to publish
        """
        if not values:
            return

        self._sequence_number += 1
        self.statistics.publish_count += 1
        self.statistics.notification_count += len(values)
        self.statistics.last_publish_time = datetime.utcnow()

        # Create notification
        notification = DataChangeNotification(
            subscription_id=self.subscription_id,
            sequence_number=self._sequence_number,
            publish_time=datetime.utcnow(),
            values=values,
        )

        # Invoke callbacks
        for callback in self._data_change_callbacks:
            try:
                await callback(notification)
            except Exception as e:
                logger.error(f"Error in data change callback: {e}")

    async def _notify_status_change(self, status: str, message: str) -> None:
        """
        Notify status change callbacks.

        Args:
            status: New status
            message: Status message
        """
        for callback in self._status_change_callbacks:
            try:
                await callback(status, message)
            except Exception as e:
                logger.error(f"Error in status change callback: {e}")

    async def _add_items_to_server(self, items: List[MonitoredItem]) -> None:
        """Add monitored items to the OPC-UA server."""
        # This would interact with the actual OPC-UA subscription
        # Implementation depends on the asyncua library
        logger.debug(f"Adding {len(items)} items to server subscription")

    async def _remove_items_from_server(self, items: List[MonitoredItem]) -> None:
        """Remove monitored items from the OPC-UA server."""
        logger.debug(f"Removing {len(items)} items from server subscription")

    async def _update_subscription_on_server(self) -> None:
        """Update subscription parameters on the server."""
        logger.debug("Updating subscription parameters on server")

    def get_monitored_item(self, node_id: str) -> Optional[MonitoredItem]:
        """
        Get a monitored item by node ID.

        Args:
            node_id: Node ID to look up

        Returns:
            MonitoredItem or None
        """
        return self._monitored_items.get(node_id)

    def get_all_monitored_items(self) -> List[MonitoredItem]:
        """Get all monitored items."""
        return list(self._monitored_items.values())

    def get_statistics(self) -> SubscriptionStatistics:
        """Get subscription statistics."""
        return self.statistics


# =============================================================================
# Subscription Manager
# =============================================================================


class SubscriptionManager:
    """
    Manages multiple OPC-UA subscriptions.

    Provides centralized management of subscriptions, including creation,
    deletion, and lifecycle management.
    """

    def __init__(self, endpoint_url: str = ""):
        """
        Initialize subscription manager.

        Args:
            endpoint_url: OPC-UA server endpoint URL
        """
        self.endpoint_url = endpoint_url
        self._subscriptions: Dict[str, OPCUASubscription] = {}
        self._lock = asyncio.Lock()

    async def create_subscription(
        self,
        config: Optional[SubscriptionConfig] = None,
        subscription_id: Optional[str] = None,
        publishing_interval_ms: int = 1000,
        nodes: Optional[List[str]] = None,
    ) -> OPCUASubscription:
        """
        Create a new subscription.

        Args:
            config: Full subscription configuration
            subscription_id: Unique subscription ID
            publishing_interval_ms: Publishing interval
            nodes: Initial nodes to monitor

        Returns:
            Created subscription
        """
        async with self._lock:
            subscription = OPCUASubscription(
                config=config,
                subscription_id=subscription_id,
                publishing_interval_ms=publishing_interval_ms,
                endpoint_url=self.endpoint_url,
            )

            # Add initial nodes
            if nodes:
                await subscription.add_monitored_items(nodes)

            self._subscriptions[subscription.subscription_id] = subscription
            logger.info(f"Created subscription: {subscription.subscription_id}")

            return subscription

    async def delete_subscription(self, subscription_id: str) -> bool:
        """
        Delete a subscription.

        Args:
            subscription_id: Subscription to delete

        Returns:
            True if subscription was deleted
        """
        async with self._lock:
            if subscription_id not in self._subscriptions:
                return False

            subscription = self._subscriptions.pop(subscription_id)
            await subscription.stop()

            logger.info(f"Deleted subscription: {subscription_id}")
            return True

    async def delete_all_subscriptions(self) -> int:
        """
        Delete all subscriptions.

        Returns:
            Number of subscriptions deleted
        """
        async with self._lock:
            count = len(self._subscriptions)

            for subscription in self._subscriptions.values():
                await subscription.stop()

            self._subscriptions.clear()
            logger.info(f"Deleted {count} subscriptions")

            return count

    def get_subscription(self, subscription_id: str) -> Optional[OPCUASubscription]:
        """
        Get a subscription by ID.

        Args:
            subscription_id: Subscription ID

        Returns:
            Subscription or None
        """
        return self._subscriptions.get(subscription_id)

    def get_all_subscriptions(self) -> List[OPCUASubscription]:
        """Get all subscriptions."""
        return list(self._subscriptions.values())

    def get_active_subscriptions(self) -> List[OPCUASubscription]:
        """Get all active subscriptions."""
        return [s for s in self._subscriptions.values() if s.is_active]

    async def start_all(self) -> None:
        """Start all subscriptions."""
        for subscription in self._subscriptions.values():
            await subscription.start()

    async def stop_all(self) -> None:
        """Stop all subscriptions."""
        for subscription in self._subscriptions.values():
            await subscription.stop()

    def get_statistics(self) -> Dict[str, SubscriptionStatistics]:
        """
        Get statistics for all subscriptions.

        Returns:
            Dictionary mapping subscription ID to statistics
        """
        return {
            sub_id: sub.get_statistics()
            for sub_id, sub in self._subscriptions.items()
        }


# =============================================================================
# Data Change Handler (for asyncua integration)
# =============================================================================


class DataChangeHandler:
    """
    Handler for asyncua data change notifications.

    This class provides the callback interface expected by the asyncua library
    for receiving data change notifications from subscriptions.
    """

    def __init__(self, subscription: OPCUASubscription):
        """
        Initialize data change handler.

        Args:
            subscription: Subscription to handle notifications for
        """
        self.subscription = subscription

    def datachange_notification(
        self,
        node: Any,
        val: Any,
        data: Any,
    ) -> None:
        """
        Handle data change notification from asyncua.

        This is the callback method called by asyncua when monitored
        items change.

        Args:
            node: OPC-UA node object
            val: New value
            data: Data change notification data
        """
        # Extract information from asyncua objects
        try:
            node_id = str(node.nodeid)

            # Extract quality from status code
            quality = OPCUAQuality.GOOD
            if hasattr(data, "monitored_item"):
                status = getattr(data.monitored_item.Value, "StatusCode", None)
                if status:
                    quality = OPCUAQuality(status.value)

            # Extract timestamps
            source_timestamp = None
            server_timestamp = None
            if hasattr(data, "monitored_item"):
                source_timestamp = getattr(data.monitored_item.Value, "SourceTimestamp", None)
                server_timestamp = getattr(data.monitored_item.Value, "ServerTimestamp", None)

            # Process asynchronously
            asyncio.create_task(
                self.subscription.process_data_change(
                    node_id=node_id,
                    value=val,
                    quality=quality,
                    source_timestamp=source_timestamp,
                    server_timestamp=server_timestamp,
                )
            )

        except Exception as e:
            logger.error(f"Error processing data change notification: {e}")

    def status_change_notification(self, status: Any) -> None:
        """
        Handle status change notification from asyncua.

        Args:
            status: New subscription status
        """
        logger.info(f"Subscription status changed: {status}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "DataChangeCallback",
    "StatusChangeCallback",
    "SubscriptionStatistics",
    # Monitored item
    "MonitoredItem",
    # Subscription
    "OPCUASubscription",
    "SubscriptionManager",
    # Handler
    "DataChangeHandler",
]
