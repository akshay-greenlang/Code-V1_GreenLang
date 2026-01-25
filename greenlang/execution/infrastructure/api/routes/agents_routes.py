"""
GreenLang Agents REST API Routes

This module provides REST API endpoints for agent management,
including listing, configuration, execution, and monitoring.

Endpoints:
    GET   /api/v1/agents           - List all agents with status
    GET   /api/v1/agents/{id}      - Get agent details
    POST  /api/v1/agents/{id}/execute - Trigger agent execution
    PATCH /api/v1/agents/{id}/config  - Update agent configuration

Features:
    - Agent lifecycle management
    - Configuration hot-reloading
    - Execution triggering with parameters
    - Real-time status monitoring
    - Capability-based filtering

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.api.routes.agents_routes import agents_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(agents_router, prefix="/api/v1")
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Depends = None
    HTTPException = Exception
    Query = None
    Request = None
    status = None
    JSONResponse = None

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================


class AgentState(str, Enum):
    """Agent lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    DEGRADED = "degraded"


class AgentCapability(str, Enum):
    """Agent capability types."""
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    CONTROL = "control"
    REPORTING = "reporting"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    ANALYSIS = "analysis"
    ORCHESTRATION = "orchestration"


class AgentType(str, Enum):
    """Types of GreenLang agents."""
    DETERMINISTIC = "deterministic"
    REASONING = "reasoning"
    INSIGHT = "insight"
    ORCHESTRATOR = "orchestrator"
    MONITOR = "monitor"


class ExecutionPriority(str, Enum):
    """Execution priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class AgentExecutionRequest(BaseModel):
    """
    Request model for agent execution.

    Attributes:
        input_data: Input data for the agent
        priority: Execution priority
        timeout_seconds: Maximum execution time
        async_execution: Whether to run asynchronously
        callback_url: URL for async result delivery
    """
    input_data: Dict[str, Any] = Field(
        ...,
        description="Input data for agent execution"
    )
    priority: ExecutionPriority = Field(
        default=ExecutionPriority.NORMAL,
        description="Execution priority level"
    )
    timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Maximum execution time in seconds"
    )
    async_execution: bool = Field(
        default=False,
        description="Run execution asynchronously"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for async result delivery"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional execution metadata"
    )

    class Config:
        schema_extra = {
            "example": {
                "input_data": {
                    "fuel_type": "natural_gas",
                    "consumption": 1000,
                    "unit": "therms"
                },
                "priority": "normal",
                "timeout_seconds": 60,
                "async_execution": False
            }
        }


class AgentConfigUpdate(BaseModel):
    """
    Request model for agent configuration update.

    Attributes:
        config: Configuration key-value pairs to update
        restart_required: Whether agent restart is needed
        validate_only: Only validate without applying
    """
    config: Dict[str, Any] = Field(
        ...,
        description="Configuration values to update"
    )
    restart_required: bool = Field(
        default=False,
        description="Whether agent restart is needed after update"
    )
    validate_only: bool = Field(
        default=False,
        description="Validate configuration without applying"
    )

    class Config:
        schema_extra = {
            "example": {
                "config": {
                    "batch_size": 100,
                    "timeout_ms": 5000,
                    "logging_level": "DEBUG"
                },
                "restart_required": False,
                "validate_only": False
            }
        }


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AgentMetrics(BaseModel):
    """
    Agent performance metrics.

    Attributes:
        total_executions: Total number of executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        avg_execution_time_ms: Average execution time in milliseconds
        last_execution_at: Timestamp of last execution
        uptime_seconds: Agent uptime in seconds
    """
    total_executions: int = Field(..., description="Total executions")
    successful_executions: int = Field(..., description="Successful executions")
    failed_executions: int = Field(..., description="Failed executions")
    avg_execution_time_ms: float = Field(..., description="Average execution time (ms)")
    last_execution_at: Optional[datetime] = Field(default=None, description="Last execution time")
    uptime_seconds: float = Field(..., description="Agent uptime in seconds")
    success_rate: float = Field(..., ge=0, le=100, description="Success rate percentage")


class AgentConfig(BaseModel):
    """
    Agent configuration details.

    Attributes:
        batch_size: Processing batch size
        timeout_ms: Operation timeout in milliseconds
        retry_attempts: Number of retry attempts
        logging_level: Logging level
        custom_settings: Custom configuration settings
    """
    batch_size: int = Field(default=50, description="Processing batch size")
    timeout_ms: int = Field(default=30000, description="Operation timeout (ms)")
    retry_attempts: int = Field(default=3, description="Retry attempts on failure")
    logging_level: str = Field(default="INFO", description="Logging level")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")


class AgentSummary(BaseModel):
    """
    Summary information about an agent.

    Attributes:
        agent_id: Unique agent identifier
        name: Human-readable agent name
        type: Agent type (deterministic, reasoning, etc.)
        state: Current agent state
        capabilities: List of agent capabilities
        version: Agent version
    """
    agent_id: str = Field(..., description="Unique agent ID")
    name: str = Field(..., description="Agent name")
    type: AgentType = Field(..., description="Agent type")
    state: AgentState = Field(..., description="Current state")
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    version: str = Field(..., description="Agent version")


class AgentDetail(BaseModel):
    """
    Detailed agent information.

    Attributes:
        agent_id: Unique agent identifier
        name: Human-readable agent name
        description: Agent description
        type: Agent type
        state: Current agent state
        capabilities: List of agent capabilities
        version: Agent version
        config: Current configuration
        metrics: Performance metrics
        dependencies: List of dependent agents/services
        created_at: Agent creation timestamp
        last_updated_at: Last update timestamp
    """
    agent_id: str = Field(..., description="Unique agent ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    type: AgentType = Field(..., description="Agent type")
    state: AgentState = Field(..., description="Current state")
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    version: str = Field(..., description="Agent version")
    config: AgentConfig = Field(..., description="Current configuration")
    metrics: AgentMetrics = Field(..., description="Performance metrics")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "agent_id": "gl-001-thermal-command",
                "name": "Thermal Command Agent",
                "description": "Process heat optimization and control agent",
                "type": "reasoning",
                "state": "ready",
                "capabilities": ["optimization", "control", "monitoring"],
                "version": "2.0.0",
                "config": {
                    "batch_size": 50,
                    "timeout_ms": 30000,
                    "retry_attempts": 3,
                    "logging_level": "INFO"
                },
                "metrics": {
                    "total_executions": 1542,
                    "successful_executions": 1520,
                    "failed_executions": 22,
                    "avg_execution_time_ms": 245.3,
                    "uptime_seconds": 86400,
                    "success_rate": 98.57
                },
                "dependencies": ["factor-broker", "event-bus"],
                "created_at": "2025-01-01T00:00:00Z",
                "last_updated_at": "2025-12-07T10:00:00Z"
            }
        }


class AgentListResponse(BaseModel):
    """
    Paginated list of agents.

    Attributes:
        items: List of agent summaries
        total: Total number of agents
        page: Current page number
        page_size: Items per page
        total_pages: Total pages
        has_next: Has next page
        has_prev: Has previous page
    """
    items: List[AgentSummary] = Field(..., description="Agent summaries")
    total: int = Field(..., description="Total agents")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class ExecutionResult(BaseModel):
    """
    Agent execution result.

    Attributes:
        execution_id: Unique execution identifier
        agent_id: Agent that performed the execution
        status: Execution status
        output: Execution output data
        execution_time_ms: Execution time in milliseconds
        started_at: Execution start time
        completed_at: Execution completion time
        error: Error details if failed
    """
    execution_id: str = Field(..., description="Unique execution ID")
    agent_id: str = Field(..., description="Agent ID")
    status: str = Field(..., description="Execution status")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Execution output")
    execution_time_ms: float = Field(..., description="Execution time (ms)")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ConfigUpdateResult(BaseModel):
    """
    Configuration update result.

    Attributes:
        success: Whether update was successful
        agent_id: Agent that was updated
        updated_fields: List of fields that were updated
        previous_values: Previous values of updated fields
        new_values: New values of updated fields
        restart_scheduled: Whether agent restart was scheduled
        validation_errors: Any validation errors
    """
    success: bool = Field(..., description="Update success status")
    agent_id: str = Field(..., description="Agent ID")
    updated_fields: List[str] = Field(..., description="Updated field names")
    previous_values: Dict[str, Any] = Field(..., description="Previous values")
    new_values: Dict[str, Any] = Field(..., description="New values")
    restart_scheduled: bool = Field(default=False, description="Restart scheduled")
    validation_errors: Optional[List[str]] = Field(default=None, description="Validation errors")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# STORAGE (In-memory for demonstration)
# =============================================================================

# Sample agents for demonstration
_agents: Dict[str, AgentDetail] = {
    "gl-001-thermal-command": AgentDetail(
        agent_id="gl-001-thermal-command",
        name="Thermal Command Agent",
        description="Process heat optimization and control agent for industrial furnaces",
        type=AgentType.REASONING,
        state=AgentState.READY,
        capabilities=[AgentCapability.OPTIMIZATION, AgentCapability.CONTROL, AgentCapability.MONITORING],
        version="2.0.0",
        config=AgentConfig(),
        metrics=AgentMetrics(
            total_executions=1542,
            successful_executions=1520,
            failed_executions=22,
            avg_execution_time_ms=245.3,
            last_execution_at=datetime.now(timezone.utc),
            uptime_seconds=86400,
            success_rate=98.57
        ),
        dependencies=["factor-broker", "event-bus"],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        last_updated_at=datetime.now(timezone.utc)
    ),
    "gl-010-emissions-guardian": AgentDetail(
        agent_id="gl-010-emissions-guardian",
        name="Emissions Guardian Agent",
        description="Real-time emissions monitoring and compliance agent",
        type=AgentType.MONITOR,
        state=AgentState.READY,
        capabilities=[AgentCapability.MONITORING, AgentCapability.VALIDATION, AgentCapability.REPORTING],
        version="1.5.0",
        config=AgentConfig(batch_size=100, timeout_ms=15000),
        metrics=AgentMetrics(
            total_executions=8432,
            successful_executions=8410,
            failed_executions=22,
            avg_execution_time_ms=89.2,
            last_execution_at=datetime.now(timezone.utc),
            uptime_seconds=172800,
            success_rate=99.74
        ),
        dependencies=["emissions-calculator", "compliance-engine"],
        created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        last_updated_at=datetime.now(timezone.utc)
    ),
    "gl-018-unified-combustion": AgentDetail(
        agent_id="gl-018-unified-combustion",
        name="Unified Combustion Agent",
        description="Comprehensive combustion analysis and optimization",
        type=AgentType.DETERMINISTIC,
        state=AgentState.READY,
        capabilities=[AgentCapability.CALCULATION, AgentCapability.ANALYSIS, AgentCapability.OPTIMIZATION],
        version="1.2.0",
        config=AgentConfig(batch_size=25, timeout_ms=60000),
        metrics=AgentMetrics(
            total_executions=956,
            successful_executions=950,
            failed_executions=6,
            avg_execution_time_ms=1245.8,
            last_execution_at=datetime.now(timezone.utc),
            uptime_seconds=259200,
            success_rate=99.37
        ),
        dependencies=["thermodynamics-engine", "factor-broker"],
        created_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
        last_updated_at=datetime.now(timezone.utc)
    )
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_execution_id() -> str:
    """Generate a unique execution ID."""
    return f"exec_{uuid.uuid4().hex[:12]}"


# =============================================================================
# ROUTER DEFINITION
# =============================================================================

if FASTAPI_AVAILABLE:
    agents_router = APIRouter(
        prefix="/agents",
        tags=["Agents"],
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            429: {"model": ErrorResponse, "description": "Rate Limited"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        }
    )


    @agents_router.get(
        "",
        response_model=AgentListResponse,
        summary="List all agents",
        description="""
        Retrieve a paginated list of all registered agents.

        Supports filtering by:
        - State (ready, running, error, etc.)
        - Capability (monitoring, optimization, etc.)
        - Type (deterministic, reasoning, insight)

        Results are sorted by agent name by default.
        """,
        operation_id="list_agents"
    )
    async def list_agents(
        request: Request,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        state: Optional[AgentState] = Query(None, description="Filter by state"),
        capability: Optional[AgentCapability] = Query(None, description="Filter by capability"),
        agent_type: Optional[AgentType] = Query(None, description="Filter by type"),
    ) -> AgentListResponse:
        """
        List all agents with pagination and filtering.

        Args:
            request: FastAPI request object
            page: Page number
            page_size: Items per page
            state: Optional state filter
            capability: Optional capability filter
            agent_type: Optional type filter

        Returns:
            Paginated list of agent summaries
        """
        logger.info(f"Listing agents: page={page}, page_size={page_size}")

        # Filter agents
        agents = list(_agents.values())

        if state:
            agents = [a for a in agents if a.state == state]

        if capability:
            agents = [a for a in agents if capability in a.capabilities]

        if agent_type:
            agents = [a for a in agents if a.type == agent_type]

        # Sort by name
        agents.sort(key=lambda x: x.name)

        # Convert to summaries
        summaries = [
            AgentSummary(
                agent_id=a.agent_id,
                name=a.name,
                type=a.type,
                state=a.state,
                capabilities=a.capabilities,
                version=a.version
            )
            for a in agents
        ]

        # Paginate
        total = len(summaries)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = summaries[start_idx:end_idx]

        return AgentListResponse(
            items=paginated,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


    @agents_router.get(
        "/{agent_id}",
        response_model=AgentDetail,
        summary="Get agent details",
        description="""
        Retrieve detailed information about a specific agent.

        Returns:
        - Agent configuration
        - Performance metrics
        - Dependencies
        - Capability information
        """,
        operation_id="get_agent"
    )
    async def get_agent(
        request: Request,
        agent_id: str,
    ) -> AgentDetail:
        """
        Get detailed information about a specific agent.

        Args:
            request: FastAPI request object
            agent_id: Agent identifier

        Returns:
            Detailed agent information

        Raises:
            HTTPException: If agent not found
        """
        logger.info(f"Getting agent details: {agent_id}")

        agent = _agents.get(agent_id)

        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "AGENT_NOT_FOUND",
                    "message": f"Agent '{agent_id}' not found"
                }
            )

        return agent


    @agents_router.post(
        "/{agent_id}/execute",
        response_model=ExecutionResult,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Trigger agent execution",
        description="""
        Trigger execution of a specific agent with provided input data.

        Supports:
        - Synchronous and asynchronous execution
        - Priority levels
        - Custom timeouts
        - Webhook callbacks for async results

        Returns execution result or job ID for async execution.
        """,
        operation_id="execute_agent"
    )
    async def execute_agent(
        request: Request,
        agent_id: str,
        execution_request: AgentExecutionRequest,
    ) -> ExecutionResult:
        """
        Trigger agent execution.

        Args:
            request: FastAPI request object
            agent_id: Agent identifier
            execution_request: Execution parameters

        Returns:
            Execution result or job reference

        Raises:
            HTTPException: If agent not found or execution fails
        """
        logger.info(f"Executing agent: {agent_id}")

        agent = _agents.get(agent_id)

        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "AGENT_NOT_FOUND",
                    "message": f"Agent '{agent_id}' not found"
                }
            )

        if agent.state not in [AgentState.READY, AgentState.RUNNING]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "AGENT_NOT_READY",
                    "message": f"Agent '{agent_id}' is in state '{agent.state.value}' and cannot accept executions"
                }
            )

        execution_id = generate_execution_id()
        started_at = datetime.now(timezone.utc)

        # Simulate execution (in production, this would call the actual agent)
        import time
        execution_time_ms = 150.0 + (len(str(execution_request.input_data)) * 0.5)

        # Simulated output
        output = {
            "input_received": execution_request.input_data,
            "agent_processed": True,
            "result": "Execution completed successfully",
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

        completed_at = datetime.now(timezone.utc)

        # Update agent metrics
        agent.metrics.total_executions += 1
        agent.metrics.successful_executions += 1
        agent.metrics.last_execution_at = completed_at

        logger.info(f"Agent execution completed: {execution_id}")

        return ExecutionResult(
            execution_id=execution_id,
            agent_id=agent_id,
            status="completed",
            output=output,
            execution_time_ms=execution_time_ms,
            started_at=started_at,
            completed_at=completed_at,
            error=None
        )


    @agents_router.patch(
        "/{agent_id}/config",
        response_model=ConfigUpdateResult,
        summary="Update agent configuration",
        description="""
        Update configuration for a specific agent.

        Supports:
        - Partial configuration updates
        - Configuration validation
        - Dry-run validation mode
        - Automatic agent restart if required

        Returns update result with previous and new values.
        """,
        operation_id="update_agent_config"
    )
    async def update_agent_config(
        request: Request,
        agent_id: str,
        config_update: AgentConfigUpdate,
    ) -> ConfigUpdateResult:
        """
        Update agent configuration.

        Args:
            request: FastAPI request object
            agent_id: Agent identifier
            config_update: Configuration update request

        Returns:
            Configuration update result

        Raises:
            HTTPException: If agent not found or validation fails
        """
        logger.info(f"Updating config for agent: {agent_id}")

        agent = _agents.get(agent_id)

        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "AGENT_NOT_FOUND",
                    "message": f"Agent '{agent_id}' not found"
                }
            )

        # Validate configuration
        validation_errors = []
        valid_fields = {"batch_size", "timeout_ms", "retry_attempts", "logging_level", "custom_settings"}

        for key in config_update.config.keys():
            if key not in valid_fields and not key.startswith("custom_"):
                validation_errors.append(f"Unknown configuration field: {key}")

        if "batch_size" in config_update.config:
            if not isinstance(config_update.config["batch_size"], int) or config_update.config["batch_size"] < 1:
                validation_errors.append("batch_size must be a positive integer")

        if "timeout_ms" in config_update.config:
            if not isinstance(config_update.config["timeout_ms"], int) or config_update.config["timeout_ms"] < 100:
                validation_errors.append("timeout_ms must be at least 100")

        if validation_errors:
            if config_update.validate_only:
                return ConfigUpdateResult(
                    success=False,
                    agent_id=agent_id,
                    updated_fields=[],
                    previous_values={},
                    new_values={},
                    restart_scheduled=False,
                    validation_errors=validation_errors
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "VALIDATION_ERROR",
                    "message": "Configuration validation failed",
                    "errors": validation_errors
                }
            )

        if config_update.validate_only:
            return ConfigUpdateResult(
                success=True,
                agent_id=agent_id,
                updated_fields=list(config_update.config.keys()),
                previous_values={},
                new_values=config_update.config,
                restart_scheduled=False,
                validation_errors=None
            )

        # Apply configuration updates
        previous_values = {}
        updated_fields = []

        for key, value in config_update.config.items():
            if hasattr(agent.config, key):
                previous_values[key] = getattr(agent.config, key)
                setattr(agent.config, key, value)
                updated_fields.append(key)
            elif key.startswith("custom_") or key == "custom_settings":
                if key == "custom_settings":
                    previous_values[key] = agent.config.custom_settings.copy()
                    agent.config.custom_settings.update(value)
                else:
                    custom_key = key.replace("custom_", "")
                    previous_values[key] = agent.config.custom_settings.get(custom_key)
                    agent.config.custom_settings[custom_key] = value
                updated_fields.append(key)

        agent.last_updated_at = datetime.now(timezone.utc)

        logger.info(f"Config updated for agent {agent_id}: {updated_fields}")

        return ConfigUpdateResult(
            success=True,
            agent_id=agent_id,
            updated_fields=updated_fields,
            previous_values=previous_values,
            new_values=config_update.config,
            restart_scheduled=config_update.restart_required,
            validation_errors=None
        )

else:
    # Provide stub router when FastAPI is not available
    agents_router = None
    logger.warning("FastAPI not available - agents_router is None")
