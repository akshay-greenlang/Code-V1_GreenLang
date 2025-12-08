"""
Agent Service for GreenLang GraphQL API

This module provides the AgentService class that manages agent lifecycle,
configuration, and monitoring for Process Heat agents.

Features:
    - Agent registry management
    - Agent status tracking
    - Configuration updates
    - Metrics collection
    - Zero-hallucination agent data access

Example:
    >>> service = AgentService()
    >>> agents = await service.get_all_agents()
    >>> agent = await service.get_agent("agent-001")
    >>> updated = await service.update_agent_config("agent-001", config)
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AgentStatusEnum(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AgentTypeEnum(str, Enum):
    """GreenLang agent types."""
    GL_001_THERMAL_COMMAND = "GL-001"
    GL_002_EMISSION_CALCULATOR = "GL-002"
    GL_003_COMPLIANCE_AUDITOR = "GL-003"
    GL_004_PREDICTIVE_MAINTENANCE = "GL-004"
    GL_005_ENERGY_OPTIMIZER = "GL-005"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    records_processed: int = 0
    processing_rate: float = 0.0
    cache_hit_ratio: float = 0.0
    error_count: int = 0


@dataclass
class AgentRecord:
    """Internal agent record for storage."""
    id: str
    name: str
    agent_type: str
    status: AgentStatusEnum
    enabled: bool
    version: str
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    metrics: AgentMetrics
    error_message: Optional[str]
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AgentConfigUpdate(BaseModel):
    """Agent configuration update input."""
    enabled: Optional[bool] = Field(None, description="Enable/disable agent")
    execution_interval_minutes: Optional[int] = Field(
        None, ge=1, le=1440, description="Execution interval in minutes"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Agent-specific parameters"
    )

    @validator('execution_interval_minutes')
    def validate_interval(cls, v: Optional[int]) -> Optional[int]:
        """Validate execution interval is reasonable."""
        if v is not None and (v < 1 or v > 1440):
            raise ValueError("Interval must be between 1 and 1440 minutes")
        return v


class AgentServiceError(Exception):
    """Base exception for agent service errors."""
    pass


class AgentNotFoundError(AgentServiceError):
    """Raised when an agent is not found."""
    pass


class AgentConfigurationError(AgentServiceError):
    """Raised when agent configuration fails."""
    pass


class AgentService:
    """
    Agent Service for managing Process Heat agents.

    Provides CRUD operations and monitoring for agents with full
    provenance tracking and audit capabilities.

    Attributes:
        _agents: In-memory agent registry (would be database in production)
        _lock: Asyncio lock for thread-safe operations

    Example:
        >>> service = AgentService()
        >>> agents = await service.get_all_agents()
        >>> for agent in agents:
        ...     print(f"{agent.id}: {agent.status}")
    """

    def __init__(self) -> None:
        """Initialize AgentService with default agent registry."""
        self._agents: Dict[str, AgentRecord] = {}
        self._lock = asyncio.Lock()
        self._initialize_default_agents()
        logger.info("AgentService initialized with %d agents", len(self._agents))

    def _initialize_default_agents(self) -> None:
        """Initialize default agent registry with sample agents."""
        now = datetime.utcnow()

        default_agents = [
            AgentRecord(
                id="agent-001",
                name="Thermal Command",
                agent_type=AgentTypeEnum.GL_001_THERMAL_COMMAND.value,
                status=AgentStatusEnum.IDLE,
                enabled=True,
                version="1.2.0",
                last_run=now,
                next_run=None,
                metrics=AgentMetrics(
                    execution_time_ms=1234.5,
                    memory_usage_mb=256.2,
                    records_processed=15000,
                    processing_rate=1250.0,
                    cache_hit_ratio=0.85,
                    error_count=0
                ),
                error_message=None,
                config={"max_batch_size": 1000, "timeout_seconds": 300},
                created_at=now,
                updated_at=now
            ),
            AgentRecord(
                id="agent-002",
                name="Emission Calculator",
                agent_type=AgentTypeEnum.GL_002_EMISSION_CALCULATOR.value,
                status=AgentStatusEnum.IDLE,
                enabled=True,
                version="1.1.0",
                last_run=now,
                next_run=None,
                metrics=AgentMetrics(
                    execution_time_ms=2456.8,
                    memory_usage_mb=384.5,
                    records_processed=25000,
                    processing_rate=980.5,
                    cache_hit_ratio=0.78,
                    error_count=1
                ),
                error_message=None,
                config={"emission_factors_version": "2024Q4"},
                created_at=now,
                updated_at=now
            ),
            AgentRecord(
                id="agent-003",
                name="Compliance Auditor",
                agent_type=AgentTypeEnum.GL_003_COMPLIANCE_AUDITOR.value,
                status=AgentStatusEnum.RUNNING,
                enabled=True,
                version="1.0.5",
                last_run=now,
                next_run=None,
                metrics=AgentMetrics(
                    execution_time_ms=5678.2,
                    memory_usage_mb=512.0,
                    records_processed=8500,
                    processing_rate=420.0,
                    cache_hit_ratio=0.92,
                    error_count=0
                ),
                error_message=None,
                config={"frameworks": ["GHG Protocol", "ISO 14064"]},
                created_at=now,
                updated_at=now
            ),
            AgentRecord(
                id="agent-004",
                name="Predictive Maintenance",
                agent_type=AgentTypeEnum.GL_004_PREDICTIVE_MAINTENANCE.value,
                status=AgentStatusEnum.PAUSED,
                enabled=False,
                version="0.9.1",
                last_run=now,
                next_run=None,
                metrics=AgentMetrics(
                    execution_time_ms=3210.0,
                    memory_usage_mb=768.3,
                    records_processed=4200,
                    processing_rate=156.2,
                    cache_hit_ratio=0.65,
                    error_count=3
                ),
                error_message="Model retraining required",
                config={"model_version": "v2.1"},
                created_at=now,
                updated_at=now
            ),
            AgentRecord(
                id="agent-005",
                name="Energy Optimizer",
                agent_type=AgentTypeEnum.GL_005_ENERGY_OPTIMIZER.value,
                status=AgentStatusEnum.IDLE,
                enabled=True,
                version="1.3.2",
                last_run=now,
                next_run=None,
                metrics=AgentMetrics(
                    execution_time_ms=1890.5,
                    memory_usage_mb=320.0,
                    records_processed=18500,
                    processing_rate=1420.5,
                    cache_hit_ratio=0.88,
                    error_count=0
                ),
                error_message=None,
                config={"optimization_target": "cost_reduction"},
                created_at=now,
                updated_at=now
            ),
        ]

        for agent in default_agents:
            self._agents[agent.id] = agent

    async def get_all_agents(
        self,
        status: Optional[str] = None
    ) -> List[AgentRecord]:
        """
        Get all agents, optionally filtered by status.

        Args:
            status: Optional status filter (idle, running, completed, failed, paused)

        Returns:
            List of agent records

        Raises:
            AgentServiceError: If retrieval fails
        """
        try:
            async with self._lock:
                agents = list(self._agents.values())

            if status:
                try:
                    status_enum = AgentStatusEnum(status.lower())
                    agents = [a for a in agents if a.status == status_enum]
                except ValueError:
                    logger.warning(f"Invalid status filter: {status}")
                    # Return empty list for invalid status
                    return []

            logger.debug(f"Retrieved {len(agents)} agents (status={status})")
            return agents

        except Exception as e:
            logger.error(f"Failed to get agents: {e}", exc_info=True)
            raise AgentServiceError(f"Failed to retrieve agents: {str(e)}") from e

    async def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """
        Get a specific agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentRecord if found, None otherwise

        Raises:
            AgentServiceError: If retrieval fails
        """
        try:
            async with self._lock:
                agent = self._agents.get(agent_id)

            if agent:
                logger.debug(f"Retrieved agent: {agent_id}")
            else:
                logger.debug(f"Agent not found: {agent_id}")

            return agent

        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}", exc_info=True)
            raise AgentServiceError(f"Failed to retrieve agent: {str(e)}") from e

    async def update_agent_config(
        self,
        agent_id: str,
        config: AgentConfigUpdate
    ) -> AgentRecord:
        """
        Update agent configuration.

        Args:
            agent_id: Agent identifier
            config: Configuration update

        Returns:
            Updated AgentRecord

        Raises:
            AgentNotFoundError: If agent is not found
            AgentConfigurationError: If configuration update fails
        """
        try:
            async with self._lock:
                if agent_id not in self._agents:
                    raise AgentNotFoundError(f"Agent not found: {agent_id}")

                agent = self._agents[agent_id]
                now = datetime.utcnow()

                # Apply configuration updates
                if config.enabled is not None:
                    agent.enabled = config.enabled
                    logger.info(f"Agent {agent_id} enabled={config.enabled}")

                if config.execution_interval_minutes is not None:
                    agent.config["execution_interval_minutes"] = config.execution_interval_minutes
                    logger.info(
                        f"Agent {agent_id} interval={config.execution_interval_minutes}m"
                    )

                if config.parameters is not None:
                    agent.config.update(config.parameters)
                    logger.info(f"Agent {agent_id} parameters updated")

                agent.updated_at = now

                # Calculate provenance hash for audit trail
                provenance_hash = self._calculate_config_hash(agent)
                logger.debug(f"Config update hash: {provenance_hash[:16]}...")

                return agent

        except AgentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}", exc_info=True)
            raise AgentConfigurationError(
                f"Failed to update agent configuration: {str(e)}"
            ) from e

    async def start_agent(self, agent_id: str) -> AgentRecord:
        """
        Start an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Updated AgentRecord

        Raises:
            AgentNotFoundError: If agent is not found
            AgentServiceError: If start fails
        """
        try:
            async with self._lock:
                if agent_id not in self._agents:
                    raise AgentNotFoundError(f"Agent not found: {agent_id}")

                agent = self._agents[agent_id]

                if not agent.enabled:
                    raise AgentServiceError(f"Agent {agent_id} is disabled")

                if agent.status == AgentStatusEnum.RUNNING:
                    logger.warning(f"Agent {agent_id} is already running")
                    return agent

                agent.status = AgentStatusEnum.RUNNING
                agent.updated_at = datetime.utcnow()
                agent.error_message = None

                logger.info(f"Agent {agent_id} started")
                return agent

        except (AgentNotFoundError, AgentServiceError):
            raise
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {e}", exc_info=True)
            raise AgentServiceError(f"Failed to start agent: {str(e)}") from e

    async def stop_agent(self, agent_id: str) -> AgentRecord:
        """
        Stop an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Updated AgentRecord

        Raises:
            AgentNotFoundError: If agent is not found
            AgentServiceError: If stop fails
        """
        try:
            async with self._lock:
                if agent_id not in self._agents:
                    raise AgentNotFoundError(f"Agent not found: {agent_id}")

                agent = self._agents[agent_id]
                agent.status = AgentStatusEnum.IDLE
                agent.last_run = datetime.utcnow()
                agent.updated_at = agent.last_run

                logger.info(f"Agent {agent_id} stopped")
                return agent

        except AgentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {e}", exc_info=True)
            raise AgentServiceError(f"Failed to stop agent: {str(e)}") from e

    async def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """
        Get agent performance metrics.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentMetrics if found, None otherwise
        """
        agent = await self.get_agent(agent_id)
        return agent.metrics if agent else None

    async def update_agent_metrics(
        self,
        agent_id: str,
        metrics: AgentMetrics
    ) -> AgentRecord:
        """
        Update agent metrics.

        Args:
            agent_id: Agent identifier
            metrics: New metrics

        Returns:
            Updated AgentRecord

        Raises:
            AgentNotFoundError: If agent is not found
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise AgentNotFoundError(f"Agent not found: {agent_id}")

            agent = self._agents[agent_id]
            agent.metrics = metrics
            agent.updated_at = datetime.utcnow()

            logger.debug(f"Agent {agent_id} metrics updated")
            return agent

    def _calculate_config_hash(self, agent: AgentRecord) -> str:
        """Calculate SHA-256 hash for agent configuration (provenance)."""
        config_str = f"{agent.id}:{agent.enabled}:{agent.config}"
        return hashlib.sha256(config_str.encode()).hexdigest()


# Singleton instance for global access
_agent_service_instance: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """
    Get the global AgentService instance.

    Returns:
        AgentService singleton instance
    """
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = AgentService()
    return _agent_service_instance


__all__ = [
    "AgentService",
    "AgentRecord",
    "AgentMetrics",
    "AgentConfigUpdate",
    "AgentStatusEnum",
    "AgentTypeEnum",
    "AgentServiceError",
    "AgentNotFoundError",
    "AgentConfigurationError",
    "get_agent_service",
]
