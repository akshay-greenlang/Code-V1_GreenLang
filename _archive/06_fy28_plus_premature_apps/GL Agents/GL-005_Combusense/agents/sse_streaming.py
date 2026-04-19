# -*- coding: utf-8 -*-
"""
Server-Sent Events (SSE) Streaming for GL-005 COMBUSENSE

Implements real-time SSE streaming for combustion diagnostics
as specified in GL-005 Playbook Section 12.3.

Event Types:
    - cqi_update: CQI score updates (1-5 second cadence)
    - anomaly_start: New anomaly incident detected
    - anomaly_update: Ongoing anomaly status update
    - anomaly_end: Anomaly incident resolved
    - safety_status: Safety boundary status changes
    - heartbeat: Connection keep-alive

Reference: GL-005 Playbook Section 12.3 (SSE Event Contract)

Author: GreenLang GL-005 Team
Version: 1.0.0
Availability Target: 99.9%
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# SSE configuration
SSE_SCHEMA_VERSION = "1.0"
HEARTBEAT_INTERVAL_SECONDS = 15
MAX_RECONNECT_TIMEOUT_SECONDS = 30
EVENT_QUEUE_SIZE = 1000


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SSEEventType(str, Enum):
    """SSE event types per playbook Section 12.3"""
    CQI_UPDATE = "cqi_update"
    ANOMALY_START = "anomaly_start"
    ANOMALY_UPDATE = "anomaly_update"
    ANOMALY_END = "anomaly_end"
    SAFETY_STATUS = "safety_status"
    HEARTBEAT = "heartbeat"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    CONFIGURATION_UPDATE = "configuration_update"


class AnomalyStatus(str, Enum):
    """Anomaly lifecycle status"""
    ACTIVE = "ACTIVE"
    UPDATING = "UPDATING"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"


# =============================================================================
# SSE EVENT MODELS
# =============================================================================

class SSEEvent(BaseModel):
    """Base SSE event structure"""
    schema_version: str = SSE_SCHEMA_VERSION
    event_type: SSEEventType
    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    asset_id: str
    data: Dict[str, Any]


class CQIUpdateEvent(BaseModel):
    """CQI update event data (Section 12.3 example)"""
    schema_version: str = SSE_SCHEMA_VERSION
    asset_id: str
    ts_utc: str
    cqi_total: float
    cqi_components: Dict[str, float]
    confidence: float
    active_incidents: List[str]
    grade: str
    is_capped: bool = False
    cap_reason: Optional[str] = None
    provenance_hash: str


class AnomalyEventData(BaseModel):
    """Anomaly event data (Appendix B example)"""
    schema_version: str = SSE_SCHEMA_VERSION
    incident_id: str
    asset_id: str
    event_type: str  # Anomaly type (CO_SPIKE, etc.)
    severity: str  # S1-S4
    confidence: float
    start_ts_utc: str
    last_update_ts_utc: str
    status: AnomalyStatus
    cqi_impact: float
    top_drivers: List[Dict[str, Any]]
    recommended_checks: List[str]
    links: Dict[str, str] = Field(default_factory=dict)


class SafetyStatusEvent(BaseModel):
    """Safety status event data"""
    schema_version: str = SSE_SCHEMA_VERSION
    asset_id: str
    ts_utc: str
    overall_status: str  # NORMAL, WARNING, CRITICAL
    interlocks_healthy: bool
    bypass_active: bool
    bypass_details: Optional[Dict[str, Any]] = None
    envelope_excursions: List[str] = Field(default_factory=list)
    trip_count_24h: int = 0
    data_integrity_issues: List[str] = Field(default_factory=list)


class HeartbeatEvent(BaseModel):
    """Heartbeat event data"""
    schema_version: str = SSE_SCHEMA_VERSION
    ts_utc: str
    server_status: str = "healthy"
    connected_assets: int = 0
    active_streams: int = 0


# =============================================================================
# SSE STREAM MANAGER
# =============================================================================

@dataclass
class SSEClient:
    """Represents a connected SSE client"""
    client_id: str
    asset_filter: Optional[Set[str]] = None  # None means all assets
    event_filter: Optional[Set[SSEEventType]] = None  # None means all events
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_id: Optional[str] = None
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=EVENT_QUEUE_SIZE))


class SSEStreamManager:
    """
    Manager for SSE streaming connections

    Handles client connections, event broadcasting, and reconnection.
    """

    def __init__(
        self,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
        max_clients: int = 1000
    ):
        """
        Initialize SSE stream manager

        Args:
            heartbeat_interval: Seconds between heartbeat events
            max_clients: Maximum concurrent SSE clients
        """
        self.heartbeat_interval = heartbeat_interval
        self.max_clients = max_clients
        self.clients: Dict[str, SSEClient] = {}
        self._event_counter = 0
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(f"SSE Stream Manager initialized (max_clients={max_clients})")

    async def start(self):
        """Start the SSE manager"""
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("SSE Stream Manager started")

    async def stop(self):
        """Stop the SSE manager"""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("SSE Stream Manager stopped")

    async def register_client(
        self,
        client_id: str,
        asset_filter: Optional[Set[str]] = None,
        event_filter: Optional[Set[SSEEventType]] = None,
        last_event_id: Optional[str] = None
    ) -> SSEClient:
        """
        Register a new SSE client

        Args:
            client_id: Unique client identifier
            asset_filter: Optional set of asset IDs to receive events for
            event_filter: Optional set of event types to receive
            last_event_id: Last event ID for reconnection

        Returns:
            SSEClient instance
        """
        async with self._lock:
            if len(self.clients) >= self.max_clients:
                raise ValueError(f"Maximum clients ({self.max_clients}) reached")

            client = SSEClient(
                client_id=client_id,
                asset_filter=asset_filter,
                event_filter=event_filter,
                last_event_id=last_event_id
            )
            self.clients[client_id] = client
            logger.info(f"SSE client registered: {client_id}")
            return client

    async def unregister_client(self, client_id: str):
        """Unregister an SSE client"""
        async with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"SSE client unregistered: {client_id}")

    async def broadcast_event(
        self,
        event_type: SSEEventType,
        asset_id: str,
        data: Dict[str, Any]
    ):
        """
        Broadcast an event to all matching clients

        Args:
            event_type: Type of SSE event
            asset_id: Asset the event relates to
            data: Event data payload
        """
        self._event_counter += 1
        event_id = f"{int(time.time())}-{asset_id}-{self._event_counter}"

        event = SSEEvent(
            event_type=event_type,
            event_id=event_id,
            asset_id=asset_id,
            data=data
        )

        async with self._lock:
            for client_id, client in self.clients.items():
                # Check asset filter
                if client.asset_filter and asset_id not in client.asset_filter:
                    continue

                # Check event type filter
                if client.event_filter and event_type not in client.event_filter:
                    continue

                # Try to add to client queue
                try:
                    client.queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for client {client_id}, dropping event")

    async def broadcast_cqi_update(
        self,
        asset_id: str,
        cqi_total: float,
        cqi_components: Dict[str, float],
        confidence: float,
        active_incidents: List[str],
        grade: str,
        is_capped: bool = False,
        cap_reason: Optional[str] = None,
        provenance_hash: str = ""
    ):
        """
        Broadcast CQI update event

        Args:
            asset_id: Asset identifier
            cqi_total: Total CQI score
            cqi_components: Component scores
            confidence: Confidence level
            active_incidents: List of active incident IDs
            grade: CQI grade (excellent, good, etc.)
            is_capped: Whether CQI is capped
            cap_reason: Reason for cap
            provenance_hash: Audit hash
        """
        event_data = CQIUpdateEvent(
            asset_id=asset_id,
            ts_utc=datetime.now(timezone.utc).isoformat(),
            cqi_total=cqi_total,
            cqi_components=cqi_components,
            confidence=confidence,
            active_incidents=active_incidents,
            grade=grade,
            is_capped=is_capped,
            cap_reason=cap_reason,
            provenance_hash=provenance_hash
        )

        await self.broadcast_event(
            SSEEventType.CQI_UPDATE,
            asset_id,
            event_data.model_dump()
        )

    async def broadcast_anomaly_start(
        self,
        incident_id: str,
        asset_id: str,
        event_type: str,
        severity: str,
        confidence: float,
        cqi_impact: float,
        top_drivers: List[Dict[str, Any]],
        recommended_checks: List[str]
    ):
        """Broadcast anomaly start event"""
        now = datetime.now(timezone.utc).isoformat()
        event_data = AnomalyEventData(
            incident_id=incident_id,
            asset_id=asset_id,
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            start_ts_utc=now,
            last_update_ts_utc=now,
            status=AnomalyStatus.ACTIVE,
            cqi_impact=cqi_impact,
            top_drivers=top_drivers,
            recommended_checks=recommended_checks
        )

        await self.broadcast_event(
            SSEEventType.ANOMALY_START,
            asset_id,
            event_data.model_dump()
        )

    async def broadcast_anomaly_update(
        self,
        incident_id: str,
        asset_id: str,
        cqi_impact: float,
        top_drivers: List[Dict[str, Any]],
        status: AnomalyStatus = AnomalyStatus.UPDATING
    ):
        """Broadcast anomaly update event"""
        event_data = {
            "incident_id": incident_id,
            "asset_id": asset_id,
            "last_update_ts_utc": datetime.now(timezone.utc).isoformat(),
            "status": status.value,
            "cqi_impact": cqi_impact,
            "top_drivers": top_drivers
        }

        await self.broadcast_event(
            SSEEventType.ANOMALY_UPDATE,
            asset_id,
            event_data
        )

    async def broadcast_anomaly_end(
        self,
        incident_id: str,
        asset_id: str,
        resolution: str = "Resolved"
    ):
        """Broadcast anomaly end event"""
        event_data = {
            "incident_id": incident_id,
            "asset_id": asset_id,
            "end_ts_utc": datetime.now(timezone.utc).isoformat(),
            "status": AnomalyStatus.RESOLVED.value,
            "resolution": resolution
        }

        await self.broadcast_event(
            SSEEventType.ANOMALY_END,
            asset_id,
            event_data
        )

    async def broadcast_safety_status(
        self,
        asset_id: str,
        overall_status: str,
        interlocks_healthy: bool,
        bypass_active: bool,
        bypass_details: Optional[Dict[str, Any]] = None,
        envelope_excursions: Optional[List[str]] = None,
        trip_count_24h: int = 0,
        data_integrity_issues: Optional[List[str]] = None
    ):
        """Broadcast safety status event"""
        event_data = SafetyStatusEvent(
            asset_id=asset_id,
            ts_utc=datetime.now(timezone.utc).isoformat(),
            overall_status=overall_status,
            interlocks_healthy=interlocks_healthy,
            bypass_active=bypass_active,
            bypass_details=bypass_details,
            envelope_excursions=envelope_excursions or [],
            trip_count_24h=trip_count_24h,
            data_integrity_issues=data_integrity_issues or []
        )

        await self.broadcast_event(
            SSEEventType.SAFETY_STATUS,
            asset_id,
            event_data.model_dump()
        )

    async def _heartbeat_loop(self):
        """Send periodic heartbeat events"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                heartbeat = HeartbeatEvent(
                    ts_utc=datetime.now(timezone.utc).isoformat(),
                    connected_assets=len(set(
                        c.asset_filter.pop() if c.asset_filter and len(c.asset_filter) == 1 else "all"
                        for c in self.clients.values()
                    )),
                    active_streams=len(self.clients)
                )

                # Send heartbeat to all clients
                async with self._lock:
                    event = SSEEvent(
                        event_type=SSEEventType.HEARTBEAT,
                        event_id=f"hb-{int(time.time())}",
                        asset_id="system",
                        data=heartbeat.model_dump()
                    )

                    for client in self.clients.values():
                        try:
                            client.queue.put_nowait(event)
                        except asyncio.QueueFull:
                            pass  # Skip heartbeat if queue full

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get SSE manager statistics"""
        return {
            "total_clients": len(self.clients),
            "total_events_sent": self._event_counter,
            "running": self._running,
            "heartbeat_interval": self.heartbeat_interval
        }


# =============================================================================
# SSE RESPONSE GENERATOR
# =============================================================================

def format_sse_event(event: SSEEvent) -> str:
    """
    Format SSE event as text/event-stream data

    Args:
        event: SSE event to format

    Returns:
        Formatted SSE string
    """
    lines = []
    lines.append(f"event: {event.event_type.value}")
    lines.append(f"id: {event.event_id}")
    lines.append(f"data: {json.dumps(event.data)}")
    lines.append("")  # Empty line to end event
    return "\n".join(lines) + "\n"


async def sse_event_generator(
    client: SSEClient,
    manager: SSEStreamManager
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events for a client

    Args:
        client: SSE client
        manager: SSE stream manager

    Yields:
        Formatted SSE event strings
    """
    try:
        while True:
            try:
                # Wait for event with timeout (for cleanup)
                event = await asyncio.wait_for(
                    client.queue.get(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS * 2
                )
                yield format_sse_event(event)
                client.last_event_id = event.event_id

            except asyncio.TimeoutError:
                # No events, continue waiting
                continue

    except asyncio.CancelledError:
        pass
    finally:
        await manager.unregister_client(client.client_id)


# =============================================================================
# FASTAPI ENDPOINT HELPERS
# =============================================================================

async def create_sse_response(
    request: Request,
    manager: SSEStreamManager,
    client_id: str,
    asset_filter: Optional[Set[str]] = None,
    event_filter: Optional[Set[SSEEventType]] = None,
    last_event_id: Optional[str] = None
) -> StreamingResponse:
    """
    Create SSE streaming response for FastAPI

    Args:
        request: FastAPI request
        manager: SSE stream manager
        client_id: Client identifier
        asset_filter: Optional asset filter
        event_filter: Optional event type filter
        last_event_id: Last event ID for reconnection

    Returns:
        StreamingResponse for SSE
    """
    client = await manager.register_client(
        client_id=client_id,
        asset_filter=asset_filter,
        event_filter=event_filter,
        last_event_id=last_event_id
    )

    return StreamingResponse(
        sse_event_generator(client, manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =============================================================================
# SSE DIAGNOSTICS STREAM
# =============================================================================

class DiagnosticsSSEStream:
    """
    High-level diagnostics SSE stream

    Integrates with CQI calculator and anomaly detector to provide
    real-time streaming diagnostics.
    """

    def __init__(self, manager: SSEStreamManager):
        """
        Initialize diagnostics stream

        Args:
            manager: SSE stream manager
        """
        self.manager = manager
        self._cqi_update_interval = 5.0  # seconds
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

    async def start(self, cqi_callback: Callable[[], Dict[str, Any]]):
        """
        Start diagnostics streaming

        Args:
            cqi_callback: Callback to get current CQI data
        """
        self._running = True
        self._update_task = asyncio.create_task(
            self._cqi_update_loop(cqi_callback)
        )
        logger.info("Diagnostics SSE stream started")

    async def stop(self):
        """Stop diagnostics streaming"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Diagnostics SSE stream stopped")

    async def _cqi_update_loop(self, cqi_callback: Callable[[], Dict[str, Any]]):
        """Periodic CQI update loop"""
        while self._running:
            try:
                await asyncio.sleep(self._cqi_update_interval)

                # Get current CQI data
                cqi_data = cqi_callback()
                if cqi_data:
                    await self.manager.broadcast_cqi_update(**cqi_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CQI update loop error: {e}")

    async def publish_anomaly(
        self,
        incident_id: str,
        asset_id: str,
        event_type: str,
        severity: str,
        confidence: float,
        cqi_impact: float,
        top_drivers: List[Dict[str, Any]],
        recommended_checks: List[str]
    ):
        """Publish new anomaly via SSE"""
        await self.manager.broadcast_anomaly_start(
            incident_id=incident_id,
            asset_id=asset_id,
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            cqi_impact=cqi_impact,
            top_drivers=top_drivers,
            recommended_checks=recommended_checks
        )

    async def update_anomaly(
        self,
        incident_id: str,
        asset_id: str,
        cqi_impact: float,
        top_drivers: List[Dict[str, Any]]
    ):
        """Update existing anomaly via SSE"""
        await self.manager.broadcast_anomaly_update(
            incident_id=incident_id,
            asset_id=asset_id,
            cqi_impact=cqi_impact,
            top_drivers=top_drivers
        )

    async def resolve_anomaly(
        self,
        incident_id: str,
        asset_id: str,
        resolution: str = "Resolved"
    ):
        """Resolve anomaly via SSE"""
        await self.manager.broadcast_anomaly_end(
            incident_id=incident_id,
            asset_id=asset_id,
            resolution=resolution
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

# Global SSE manager instance
_sse_manager: Optional[SSEStreamManager] = None


def get_sse_manager() -> SSEStreamManager:
    """Get or create the global SSE manager"""
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEStreamManager()
    return _sse_manager


async def initialize_sse():
    """Initialize and start the global SSE manager"""
    manager = get_sse_manager()
    await manager.start()
    return manager


async def shutdown_sse():
    """Shutdown the global SSE manager"""
    global _sse_manager
    if _sse_manager:
        await _sse_manager.stop()
        _sse_manager = None
