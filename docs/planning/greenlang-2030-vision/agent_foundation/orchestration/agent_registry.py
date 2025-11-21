# -*- coding: utf-8 -*-
"""
AgentRegistry - Agent discovery and registry service.

This module implements a distributed agent registry for service discovery,
capability matching, and dynamic agent registration supporting 10,000+ agents.

Example:
    >>> registry = AgentRegistry(message_bus)
    >>> await registry.initialize()
    >>>
    >>> # Register agent
    >>> descriptor = AgentDescriptor(
    ...     agent_id="agent-calculator-001",
    ...     agent_type="CalculatorAgent",
    ...     capabilities=["carbon_calculation", "scope3_emissions"],
    ...     endpoint="tcp://agent-001:5000"
    ... )
    >>> await registry.register(descriptor)
    >>>
    >>> # Discover agents by capability
    >>> agents = await registry.discover(capabilities=["carbon_calculation"])
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import logging
from datetime import datetime, timezone, timedelta
import uuid
import json
import hashlib
from collections import defaultdict
import re

from prometheus_client import Counter, Gauge, Histogram

from .message_bus import MessageBus, Message, MessageType, Priority
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)

# Metrics
registry_agents_gauge = Gauge('registry_agents_total', 'Total registered agents', ['status'])
registry_queries_counter = Counter('registry_queries_total', 'Registry queries', ['query_type'])
registry_discovery_histogram = Histogram('registry_discovery_latency_ms', 'Discovery latency')
registry_health_gauge = Gauge('registry_agent_health', 'Agent health status', ['agent_id'])


class AgentStatus(str, Enum):
    """Agent registration status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEGRADED = "DEGRADED"
    MAINTENANCE = "MAINTENANCE"
    DECOMMISSIONED = "DECOMMISSIONED"


class ServiceType(str, Enum):
    """Agent service types."""
    COMPUTATION = "COMPUTATION"
    VALIDATION = "VALIDATION"
    INTEGRATION = "INTEGRATION"
    REPORTING = "REPORTING"
    MONITORING = "MONITORING"
    COORDINATION = "COORDINATION"
    STORAGE = "STORAGE"
    ANALYSIS = "ANALYSIS"


class AgentDescriptor(BaseModel):
    """Agent service descriptor."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent implementation type")
    version: str = Field(default="1.0.0", description="Agent version")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    service_types: List[ServiceType] = Field(default_factory=list, description="Service types")
    endpoint: Optional[str] = Field(None, description="Agent endpoint URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Operational constraints")
    sla: Dict[str, float] = Field(default_factory=dict, description="SLA metrics")
    dependencies: List[str] = Field(default_factory=list, description="Required services")

    @validator('agent_id')
    def validate_agent_id(cls, v):
        """Validate agent ID format."""
        if not re.match(r'^agent-[\w-]+$', v):
            raise ValueError(f"Invalid agent ID format: {v}")
        return v

    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version."""
        if not re.match(r'^\d+\.\d+\.\d+(-[\w\.]+)?$', v):
            raise ValueError(f"Invalid version format: {v}")
        return v


class AgentRegistration(BaseModel):
    """Agent registration record."""

    descriptor: AgentDescriptor = Field(..., description="Agent descriptor")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE)
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    instances: int = Field(default=1, ge=1, description="Number of instances")
    location: Optional[str] = Field(None, description="Geographic/cluster location")
    provenance_hash: Optional[str] = Field(None, description="Registration provenance")


class DiscoveryQuery(BaseModel):
    """Service discovery query."""

    query_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    capabilities: Optional[List[str]] = Field(None, description="Required capabilities")
    service_types: Optional[List[ServiceType]] = Field(None, description="Service types")
    tags: Optional[List[str]] = Field(None, description="Required tags")
    version_constraint: Optional[str] = Field(None, description="Version requirement")
    location_preference: Optional[str] = Field(None, description="Preferred location")
    min_health_score: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)
    include_inactive: bool = Field(default=False)


class ServiceDiscovery:
    """
    Service discovery implementation.

    Provides capability-based agent discovery with health monitoring.
    """

    def __init__(self):
        """Initialize service discovery."""
        self.cache: Dict[str, Tuple[List[AgentRegistration], float]] = {}
        self.cache_ttl_seconds = 30

    async def discover(
        self,
        registry: "AgentRegistry",
        query: DiscoveryQuery
    ) -> List[AgentRegistration]:
        """
        Discover agents matching query criteria.

        Args:
            registry: Agent registry instance
            query: Discovery query

        Returns:
            List of matching agent registrations
        """
        start_time = datetime.now(timezone.utc)

        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            results, timestamp = self.cache[cache_key]
            if (datetime.now(timezone.utc).timestamp() - timestamp) < self.cache_ttl_seconds:
                return results

        # Get all registrations
        all_agents = list(registry.registrations.values())

        # Filter by status
        if not query.include_inactive:
            all_agents = [a for a in all_agents if a.status == AgentStatus.ACTIVE]

        # Filter by capabilities
        if query.capabilities:
            all_agents = [
                a for a in all_agents
                if all(cap in a.descriptor.capabilities for cap in query.capabilities)
            ]

        # Filter by service types
        if query.service_types:
            all_agents = [
                a for a in all_agents
                if any(st in a.descriptor.service_types for st in query.service_types)
            ]

        # Filter by tags
        if query.tags:
            all_agents = [
                a for a in all_agents
                if any(tag in a.descriptor.tags for tag in query.tags)
            ]

        # Filter by version
        if query.version_constraint:
            all_agents = self._filter_by_version(all_agents, query.version_constraint)

        # Filter by health score
        all_agents = [a for a in all_agents if a.health_score >= query.min_health_score]

        # Filter by location preference
        if query.location_preference:
            # Prioritize agents in preferred location
            preferred = [a for a in all_agents if a.location == query.location_preference]
            others = [a for a in all_agents if a.location != query.location_preference]
            all_agents = preferred + others

        # Sort by health score and limit results
        all_agents.sort(key=lambda a: a.health_score, reverse=True)
        results = all_agents[:query.max_results]

        # Update cache
        self.cache[cache_key] = (results, datetime.now(timezone.utc).timestamp())

        # Update metrics
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        registry_discovery_histogram.observe(duration_ms)
        registry_queries_counter.labels(query_type="discovery").inc()

        return results

    def _filter_by_version(
        self,
        agents: List[AgentRegistration],
        constraint: str
    ) -> List[AgentRegistration]:
        """Filter agents by version constraint."""
        # Simple version matching (could be enhanced with semver)
        if constraint.startswith(">="):
            min_version = constraint[2:].strip()
            return [a for a in agents if a.descriptor.version >= min_version]
        elif constraint.startswith(">"):
            min_version = constraint[1:].strip()
            return [a for a in agents if a.descriptor.version > min_version]
        elif constraint.startswith("="):
            exact_version = constraint[1:].strip()
            return [a for a in agents if a.descriptor.version == exact_version]
        else:
            return agents

    def _get_cache_key(self, query: DiscoveryQuery) -> str:
        """Generate cache key for query."""
        key_data = {
            "capabilities": sorted(query.capabilities or []),
            "service_types": sorted([st.value for st in query.service_types or []]),
            "tags": sorted(query.tags or []),
            "version": query.version_constraint,
            "location": query.location_preference,
            "health": query.min_health_score
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


class AgentRegistry:
    """
    Distributed agent registry for service discovery.

    Manages agent registration, health monitoring, and capability-based
    discovery for 10,000+ agents.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        heartbeat_interval_seconds: int = 30,
        health_check_interval_seconds: int = 60,
        deregistration_timeout_seconds: int = 300
    ):
        """Initialize agent registry."""
        self.message_bus = message_bus
        self.heartbeat_interval = heartbeat_interval_seconds
        self.health_check_interval = health_check_interval_seconds
        self.deregistration_timeout = deregistration_timeout_seconds

        # Registry storage
        self.registrations: Dict[str, AgentRegistration] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)
        self.service_type_index: Dict[ServiceType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Service discovery
        self.discovery = ServiceDiscovery()

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize registry and start background tasks."""
        logger.info("Initializing AgentRegistry")

        self._running = True

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._heartbeat_monitor())
        )
        self._tasks.append(
            asyncio.create_task(self._health_checker())
        )
        self._tasks.append(
            asyncio.create_task(self._message_handler())
        )

        logger.info(f"AgentRegistry initialized")

    async def shutdown(self) -> None:
        """Shutdown registry."""
        logger.info("Shutting down AgentRegistry")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("AgentRegistry shutdown complete")

    async def register(
        self,
        descriptor: AgentDescriptor,
        location: Optional[str] = None
    ) -> bool:
        """
        Register agent with registry.

        Args:
            descriptor: Agent descriptor
            location: Optional location/cluster identifier

        Returns:
            Success status
        """
        try:
            # Check if already registered
            if descriptor.agent_id in self.registrations:
                logger.warning(f"Agent {descriptor.agent_id} already registered")
                return await self.update(descriptor)

            # Create registration
            registration = AgentRegistration(
                descriptor=descriptor,
                location=location
            )

            # Calculate provenance
            registration.provenance_hash = self._calculate_provenance(descriptor)

            # Store registration
            self.registrations[descriptor.agent_id] = registration

            # Update indices
            self._update_indices(descriptor, add=True)

            # Update metrics
            registry_agents_gauge.labels(status=AgentStatus.ACTIVE).inc()

            # Broadcast registration event
            await self._broadcast_event("AGENT_REGISTERED", descriptor.agent_id)

            logger.info(f"Registered agent {descriptor.agent_id} (type: {descriptor.agent_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {descriptor.agent_id}: {e}")
            return False

    async def deregister(self, agent_id: str) -> bool:
        """
        Deregister agent from registry.

        Args:
            agent_id: Agent identifier

        Returns:
            Success status
        """
        if agent_id not in self.registrations:
            logger.warning(f"Agent {agent_id} not found in registry")
            return False

        try:
            registration = self.registrations[agent_id]

            # Update indices
            self._update_indices(registration.descriptor, add=False)

            # Remove registration
            del self.registrations[agent_id]

            # Update metrics
            registry_agents_gauge.labels(status=registration.status).dec()

            # Broadcast deregistration event
            await self._broadcast_event("AGENT_DEREGISTERED", agent_id)

            logger.info(f"Deregistered agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {e}")
            return False

    async def update(self, descriptor: AgentDescriptor) -> bool:
        """
        Update agent registration.

        Args:
            descriptor: Updated agent descriptor

        Returns:
            Success status
        """
        if descriptor.agent_id not in self.registrations:
            logger.warning(f"Agent {descriptor.agent_id} not registered")
            return False

        try:
            registration = self.registrations[descriptor.agent_id]

            # Update indices
            self._update_indices(registration.descriptor, add=False)
            self._update_indices(descriptor, add=True)

            # Update descriptor
            registration.descriptor = descriptor
            registration.last_heartbeat = datetime.now(timezone.utc).isoformat()

            # Recalculate provenance
            registration.provenance_hash = self._calculate_provenance(descriptor)

            # Broadcast update event
            await self._broadcast_event("AGENT_UPDATED", descriptor.agent_id)

            logger.debug(f"Updated agent {descriptor.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update agent {descriptor.agent_id}: {e}")
            return False

    async def heartbeat(
        self,
        agent_id: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Record agent heartbeat.

        Args:
            agent_id: Agent identifier
            metrics: Optional performance metrics

        Returns:
            Success status
        """
        if agent_id not in self.registrations:
            logger.warning(f"Heartbeat from unregistered agent {agent_id}")
            return False

        registration = self.registrations[agent_id]
        registration.last_heartbeat = datetime.now(timezone.utc).isoformat()

        if metrics:
            registration.metrics.update(metrics)

        # Update health score based on metrics
        registration.health_score = self._calculate_health_score(registration)

        # Update status if degraded
        if registration.health_score < 0.5 and registration.status == AgentStatus.ACTIVE:
            registration.status = AgentStatus.DEGRADED
            logger.warning(f"Agent {agent_id} degraded (health: {registration.health_score:.2f})")

        return True

    async def discover(
        self,
        capabilities: Optional[List[str]] = None,
        service_types: Optional[List[ServiceType]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[AgentRegistration]:
        """
        Discover agents by capabilities.

        Args:
            capabilities: Required capabilities
            service_types: Service types
            tags: Required tags
            **kwargs: Additional query parameters

        Returns:
            List of matching agent registrations
        """
        query = DiscoveryQuery(
            capabilities=capabilities,
            service_types=service_types,
            tags=tags,
            **kwargs
        )

        return await self.discovery.discover(self, query)

    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID."""
        return self.registrations.get(agent_id)

    async def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[str] = None
    ) -> List[AgentRegistration]:
        """List registered agents with optional filters."""
        agents = list(self.registrations.values())

        if status:
            agents = [a for a in agents if a.status == status]

        if agent_type:
            agents = [a for a in agents if a.descriptor.agent_type == agent_type]

        return agents

    def _update_indices(self, descriptor: AgentDescriptor, add: bool = True) -> None:
        """Update registry indices."""
        # Capability index
        for capability in descriptor.capabilities:
            if add:
                self.capability_index[capability].add(descriptor.agent_id)
            else:
                self.capability_index[capability].discard(descriptor.agent_id)

        # Service type index
        for service_type in descriptor.service_types:
            if add:
                self.service_type_index[service_type].add(descriptor.agent_id)
            else:
                self.service_type_index[service_type].discard(descriptor.agent_id)

        # Tag index
        for tag in descriptor.tags:
            if add:
                self.tag_index[tag].add(descriptor.agent_id)
            else:
                self.tag_index[tag].discard(descriptor.agent_id)

    def _calculate_provenance(self, descriptor: AgentDescriptor) -> str:
        """Calculate provenance hash for registration."""
        data = {
            "agent_id": descriptor.agent_id,
            "agent_type": descriptor.agent_type,
            "version": descriptor.version,
            "capabilities": sorted(descriptor.capabilities),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _calculate_health_score(self, registration: AgentRegistration) -> float:
        """Calculate agent health score."""
        score = 1.0

        # Check heartbeat recency
        last_heartbeat = datetime.fromisoformat(registration.last_heartbeat)
        heartbeat_age = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()

        if heartbeat_age > self.heartbeat_interval * 2:
            score *= 0.5
        elif heartbeat_age > self.heartbeat_interval:
            score *= 0.8

        # Check metrics
        if registration.metrics:
            # Error rate
            error_rate = registration.metrics.get("error_rate", 0)
            score *= (1 - min(error_rate, 0.5))

            # Response time
            response_time = registration.metrics.get("response_time_ms", 0)
            if response_time > 5000:
                score *= 0.5
            elif response_time > 2000:
                score *= 0.8

            # CPU usage
            cpu_usage = registration.metrics.get("cpu_usage", 0)
            if cpu_usage > 0.9:
                score *= 0.5
            elif cpu_usage > 0.7:
                score *= 0.8

        return max(0.0, min(1.0, score))

    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                timeout_threshold = now - timedelta(seconds=self.deregistration_timeout)

                for agent_id, registration in list(self.registrations.items()):
                    last_heartbeat = datetime.fromisoformat(registration.last_heartbeat)

                    # Check for timeout
                    if last_heartbeat < timeout_threshold:
                        logger.warning(f"Agent {agent_id} timed out - deregistering")
                        registration.status = AgentStatus.DECOMMISSIONED
                        await self.deregister(agent_id)
                    elif registration.status == AgentStatus.ACTIVE:
                        # Check if becoming inactive
                        inactive_threshold = now - timedelta(seconds=self.heartbeat_interval * 3)
                        if last_heartbeat < inactive_threshold:
                            registration.status = AgentStatus.INACTIVE
                            logger.info(f"Agent {agent_id} marked inactive")

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")

    async def _health_checker(self) -> None:
        """Perform health checks on agents."""
        while self._running:
            try:
                for agent_id, registration in self.registrations.items():
                    # Calculate health score
                    old_score = registration.health_score
                    registration.health_score = self._calculate_health_score(registration)

                    # Update metrics
                    registry_health_gauge.labels(agent_id=agent_id).set(registration.health_score)

                    # Check for significant change
                    if abs(old_score - registration.health_score) > 0.2:
                        logger.info(
                            f"Agent {agent_id} health changed: {old_score:.2f} -> {registration.health_score:.2f}"
                        )

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health checker error: {e}")

    async def _message_handler(self) -> None:
        """Handle registry-related messages."""
        try:
            async for message in self.message_bus.subscribe(["agent.lifecycle"]):
                if message.payload.get("event") == "HEARTBEAT":
                    agent_id = message.sender_id
                    metrics = message.payload.get("metrics")
                    await self.heartbeat(agent_id, metrics)

        except Exception as e:
            logger.error(f"Message handler error: {e}")

    async def _broadcast_event(self, event_type: str, agent_id: str) -> None:
        """Broadcast registry event."""
        message = Message(
            sender_id="registry",
            recipient_id="broadcast",
            message_type=MessageType.EVENT,
            payload={
                "event": event_type,
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        await self.message_bus.publish(message, topic="agent.lifecycle")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics."""
        status_distribution = defaultdict(int)
        for registration in self.registrations.values():
            status_distribution[registration.status] += 1

        avg_health = 0
        if self.registrations:
            avg_health = sum(r.health_score for r in self.registrations.values()) / len(self.registrations)

        return {
            "total_agents": len(self.registrations),
            "status_distribution": dict(status_distribution),
            "average_health_score": avg_health,
            "unique_capabilities": len(self.capability_index),
            "unique_tags": len(self.tag_index)
        }