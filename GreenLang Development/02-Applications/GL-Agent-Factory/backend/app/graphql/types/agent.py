"""
Process Heat Agent GraphQL Types

Defines GraphQL types for all 143 Process Heat agents in the GreenLang registry.
Supports queries, filtering, and real-time status tracking.

Example:
    query {
        agent(id: "GL-022") {
            id
            name
            category
            status
            healthScore
        }
    }
"""

import strawberry
from strawberry.scalars import JSON
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


# =============================================================================
# Custom Scalars
# =============================================================================


@strawberry.scalar(
    description="Date and time in ISO 8601 format",
    serialize=lambda v: v.isoformat() if v else None,
    parse_value=lambda v: datetime.fromisoformat(v) if v else None,
)
class DateTime:
    """Custom DateTime scalar for ISO 8601 format."""
    pass


# =============================================================================
# Enums
# =============================================================================


@strawberry.enum
class AgentStatusEnum(Enum):
    """Agent operational status."""

    AVAILABLE = "available"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    INITIALIZING = "initializing"


@strawberry.enum
class AgentCategoryEnum(Enum):
    """Process Heat agent categories aligned with registry."""

    # Climate & Compliance
    EMISSIONS = "Emissions"
    COMPLIANCE = "Compliance"
    REPORTING = "Reporting"
    CARBON = "Carbon"
    RISK = "Risk"

    # Process Heat
    HEAT_RECOVERY = "Heat Recovery"
    MAINTENANCE = "Maintenance"
    STEAM_SYSTEMS = "Steam Systems"
    OPTIMIZATION = "Optimization"
    COGENERATION = "Cogeneration"
    COMBUSTION = "Combustion"
    FURNACES = "Furnaces"
    FUEL_SYSTEMS = "Fuel Systems"
    PROCESS_INTEGRATION = "Process Integration"

    # Energy & Sustainability
    ENERGY_STORAGE = "Energy Storage"
    HEAT_NETWORKS = "Heat Networks"
    DECARBONIZATION = "Decarbonization"
    FUTURE_FUELS = "Future Fuels"
    RENEWABLE_HEAT = "Renewable Heat"
    HEAT_PUMPS = "Heat Pumps"
    SAFETY = "Safety"
    VISUALIZATION = "Visualization"
    OPERATIONS = "Operations"

    # Analytics
    ANALYTICS = "Analytics"
    PROCESS = "Process"
    MOTORS = "Motors"
    HEAT_TRACING = "Heat Tracing"
    EMISSIONS_CONTROL = "Emissions Control"

    # Digital & Financial
    DIGITAL_TWIN = "Digital Twin"
    SIMULATION = "Simulation"
    FINANCIAL = "Financial"
    SUSTAINABILITY = "Sustainability"
    PLANNING = "Planning"
    TRAINING = "Training"
    INVENTORY = "Inventory"
    PROCUREMENT = "Procurement"

    # Advanced
    GRID_INTEGRATION = "Grid Integration"
    RENEWABLE_ENERGY = "Renewable Energy"
    HYDROGEN = "Hydrogen"
    CARBON_CAPTURE = "Carbon Capture"
    SUPPLY_CHAIN = "Supply Chain"
    QUALITY = "Quality"
    SECURITY = "Security"
    DATA = "Data"
    INNOVATION = "Innovation"
    KNOWLEDGE = "Knowledge"
    INTEGRATION = "Integration"
    HR = "HR"
    BUILDING = "Building"
    WATER = "Water"


@strawberry.enum
class AgentTypeEnum(Enum):
    """Agent functional types aligned with registry."""

    CALCULATOR = "Calculator"
    MONITOR = "Monitor"
    REPORTER = "Reporter"
    ANALYZER = "Analyzer"
    OPTIMIZER = "Optimizer"
    CONTROLLER = "Controller"
    PREDICTOR = "Predictor"
    INTEGRATOR = "Integrator"
    VALIDATOR = "Validator"
    MANAGER = "Manager"
    TRACKER = "Tracker"
    COORDINATOR = "Coordinator"
    AUDITOR = "Auditor"
    ADVISOR = "Advisor"
    SIMULATOR = "Simulator"
    PLANNER = "Planner"
    MODELER = "Modeler"
    SCANNER = "Scanner"
    AUTOMATOR = "Automator"
    BUILDER = "Builder"
    EVALUATOR = "Evaluator"
    ASSESSOR = "Assessor"
    TRADER = "Trader"
    DRIVER = "Driver"


@strawberry.enum
class AgentPriorityEnum(Enum):
    """Agent implementation priority."""

    P0 = "P0"  # Critical - Must have
    P1 = "P1"  # High - Should have
    P2 = "P2"  # Medium - Could have
    P3 = "P3"  # Low - Nice to have


@strawberry.enum
class AgentComplexityEnum(Enum):
    """Agent implementation complexity."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


@strawberry.enum
class HealthStatusLevel(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# Input Types
# =============================================================================


@strawberry.input
class AgentFilterInput:
    """Filters for querying agents."""

    category: Optional[str] = strawberry.field(
        default=None,
        description="Filter by category (e.g., 'Steam Systems')"
    )
    agent_type: Optional[str] = strawberry.field(
        default=None,
        description="Filter by type (e.g., 'Optimizer')"
    )
    priority: Optional[str] = strawberry.field(
        default=None,
        description="Filter by priority (P0, P1, P2, P3)"
    )
    status: Optional[AgentStatusEnum] = strawberry.field(
        default=None,
        description="Filter by operational status"
    )
    search: Optional[str] = strawberry.field(
        default=None,
        description="Full-text search in name and description"
    )
    complexity: Optional[str] = strawberry.field(
        default=None,
        description="Filter by complexity (Low, Medium, High)"
    )


@strawberry.input
class AgentConfigInput:
    """Input for configuring an agent."""

    parameters: JSON = strawberry.field(
        description="Agent-specific configuration parameters"
    )
    timeout_seconds: Optional[int] = strawberry.field(
        default=300,
        description="Execution timeout in seconds"
    )
    retry_count: Optional[int] = strawberry.field(
        default=3,
        description="Number of retries on failure"
    )
    cache_enabled: Optional[bool] = strawberry.field(
        default=True,
        description="Enable result caching"
    )


# =============================================================================
# Object Types
# =============================================================================


@strawberry.type
class AgentInfoType:
    """
    Lightweight agent information from registry.

    Maps directly to AgentInfo dataclass from agents/registry.py
    """

    agent_id: str = strawberry.field(description="Unique agent ID (e.g., GL-022)")
    agent_name: str = strawberry.field(description="Agent name (e.g., SUPERHEAT-CTRL)")
    module_path: str = strawberry.field(description="Python module path")
    class_name: str = strawberry.field(description="Python class name")
    category: str = strawberry.field(description="Agent category")
    agent_type: str = strawberry.field(description="Agent functional type")
    complexity: str = strawberry.field(description="Implementation complexity")
    priority: str = strawberry.field(description="Implementation priority")
    market_size: Optional[str] = strawberry.field(
        default=None,
        description="Total addressable market size"
    )
    description: Optional[str] = strawberry.field(
        default=None,
        description="Agent description"
    )
    standards: List[str] = strawberry.field(
        default_factory=list,
        description="Applicable regulatory standards"
    )
    status: str = strawberry.field(
        default="Implemented",
        description="Implementation status"
    )


@strawberry.type
class HealthStatusType:
    """Agent health status with detailed metrics."""

    level: HealthStatusLevel = strawberry.field(
        description="Overall health level"
    )
    score: float = strawberry.field(
        description="Health score (0-100)"
    )
    last_check: DateTime = strawberry.field(
        description="Last health check timestamp"
    )
    response_time_ms: float = strawberry.field(
        description="Last response time in milliseconds"
    )
    error_rate: float = strawberry.field(
        description="Error rate (0-1) over last hour"
    )
    availability: float = strawberry.field(
        description="Availability percentage over last 24 hours"
    )
    message: Optional[str] = strawberry.field(
        default=None,
        description="Health status message"
    )
    issues: List[str] = strawberry.field(
        default_factory=list,
        description="Active health issues"
    )


@strawberry.type
class AgentMetricsType:
    """Runtime metrics for an agent."""

    total_invocations: int = strawberry.field(
        description="Total invocation count"
    )
    successful_invocations: int = strawberry.field(
        description="Successful invocation count"
    )
    failed_invocations: int = strawberry.field(
        description="Failed invocation count"
    )
    average_duration_ms: float = strawberry.field(
        description="Average execution duration in milliseconds"
    )
    p50_duration_ms: float = strawberry.field(
        description="50th percentile duration"
    )
    p95_duration_ms: float = strawberry.field(
        description="95th percentile duration"
    )
    p99_duration_ms: float = strawberry.field(
        description="99th percentile duration"
    )
    last_invocation: Optional[DateTime] = strawberry.field(
        default=None,
        description="Last invocation timestamp"
    )
    total_compute_cost_usd: float = strawberry.field(
        description="Total compute cost in USD"
    )
    cache_hit_rate: float = strawberry.field(
        description="Cache hit rate (0-1)"
    )


@strawberry.type
class AgentConfigType:
    """Agent configuration."""

    agent_id: str = strawberry.field(description="Agent ID")
    parameters: JSON = strawberry.field(description="Configuration parameters")
    timeout_seconds: int = strawberry.field(description="Execution timeout")
    retry_count: int = strawberry.field(description="Retry count on failure")
    cache_enabled: bool = strawberry.field(description="Caching enabled")
    updated_at: DateTime = strawberry.field(description="Last update timestamp")
    updated_by: Optional[str] = strawberry.field(
        default=None,
        description="User who last updated config"
    )


@strawberry.type
class ProcessHeatAgentType:
    """
    Full Process Heat agent type with all fields.

    Represents a single agent from the 143 GL Process Heat agents.
    Includes metadata, status, health, metrics, and configuration.
    """

    # Core identification
    id: str = strawberry.field(description="Internal database ID")
    agent_id: str = strawberry.field(description="Registry ID (e.g., GL-022)")
    name: str = strawberry.field(description="Human-readable name")

    # Classification
    category: str = strawberry.field(description="Agent category")
    type: str = strawberry.field(description="Agent functional type")
    complexity: str = strawberry.field(description="Implementation complexity")
    priority: str = strawberry.field(description="Implementation priority (P0-P3)")

    # Status
    status: AgentStatusEnum = strawberry.field(description="Current operational status")
    health_score: float = strawberry.field(description="Health score (0-100)")
    last_run: Optional[DateTime] = strawberry.field(
        default=None,
        description="Last execution timestamp"
    )

    # Metadata
    description: Optional[str] = strawberry.field(
        default=None,
        description="Agent description"
    )
    market_size: Optional[str] = strawberry.field(
        default=None,
        description="Total addressable market"
    )
    standards: List[str] = strawberry.field(
        default_factory=list,
        description="Applicable standards"
    )
    tags: List[str] = strawberry.field(
        default_factory=list,
        description="Searchable tags"
    )

    # Technical details
    module_path: str = strawberry.field(description="Python module path")
    class_name: str = strawberry.field(description="Python class name")
    version: str = strawberry.field(default="1.0.0", description="Agent version")
    deterministic: bool = strawberry.field(
        default=True,
        description="Whether agent produces deterministic results"
    )

    # Timestamps
    created_at: DateTime = strawberry.field(description="Creation timestamp")
    updated_at: DateTime = strawberry.field(description="Last update timestamp")

    @strawberry.field(description="Get agent health status")
    async def health(self) -> HealthStatusType:
        """Get detailed health status for this agent."""
        from datetime import datetime, timezone

        # Default health status - would be populated from monitoring service
        return HealthStatusType(
            level=HealthStatusLevel.HEALTHY if self.health_score > 80 else (
                HealthStatusLevel.DEGRADED if self.health_score > 50 else HealthStatusLevel.UNHEALTHY
            ),
            score=self.health_score,
            last_check=datetime.now(timezone.utc),
            response_time_ms=150.0,
            error_rate=0.01,
            availability=99.9,
            message="Agent operating normally" if self.health_score > 80 else "Agent experiencing issues",
            issues=[]
        )

    @strawberry.field(description="Get agent runtime metrics")
    async def metrics(self, period_days: int = 30) -> AgentMetricsType:
        """Get runtime metrics for this agent."""
        from datetime import datetime, timezone

        # Default metrics - would be populated from metrics service
        return AgentMetricsType(
            total_invocations=1000,
            successful_invocations=990,
            failed_invocations=10,
            average_duration_ms=250.0,
            p50_duration_ms=200.0,
            p95_duration_ms=500.0,
            p99_duration_ms=800.0,
            last_invocation=self.last_run,
            total_compute_cost_usd=15.50,
            cache_hit_rate=0.75
        )

    @strawberry.field(description="Get agent configuration")
    async def config(self) -> Optional[AgentConfigType]:
        """Get current configuration for this agent."""
        from datetime import datetime, timezone

        return AgentConfigType(
            agent_id=self.agent_id,
            parameters={},
            timeout_seconds=300,
            retry_count=3,
            cache_enabled=True,
            updated_at=datetime.now(timezone.utc),
            updated_by=None
        )

    @classmethod
    def from_registry_info(
        cls,
        info: "AgentInfo",
        db_id: Optional[str] = None,
        status: AgentStatusEnum = AgentStatusEnum.AVAILABLE,
        health_score: float = 100.0,
        last_run: Optional[datetime] = None,
    ) -> "ProcessHeatAgentType":
        """
        Create ProcessHeatAgentType from registry AgentInfo.

        Args:
            info: AgentInfo from registry
            db_id: Optional database ID
            status: Current operational status
            health_score: Current health score
            last_run: Last execution timestamp

        Returns:
            ProcessHeatAgentType instance
        """
        from datetime import datetime, timezone
        import uuid

        return cls(
            id=db_id or str(uuid.uuid4()),
            agent_id=info.agent_id,
            name=info.agent_name,
            category=info.category,
            type=info.agent_type,
            complexity=info.complexity,
            priority=info.priority,
            status=status,
            health_score=health_score,
            last_run=last_run,
            description=info.description or f"GreenLang {info.agent_name} agent for {info.category}",
            market_size=info.market_size,
            standards=info.standards,
            tags=[info.category, info.agent_type, info.priority],
            module_path=info.module_path,
            class_name=info.class_name,
            version="1.0.0",
            deterministic=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


# =============================================================================
# Connection Types (Cursor-based Pagination)
# =============================================================================


@strawberry.type
class PageInfo:
    """Pagination information for connections."""

    has_next_page: bool = strawberry.field(description="More pages available after")
    has_previous_page: bool = strawberry.field(description="More pages available before")
    start_cursor: Optional[str] = strawberry.field(description="Cursor of first item")
    end_cursor: Optional[str] = strawberry.field(description="Cursor of last item")
    total_count: int = strawberry.field(description="Total number of items")


@strawberry.type
class AgentEdge:
    """Edge in agent connection."""

    cursor: str = strawberry.field(description="Cursor for this item")
    node: ProcessHeatAgentType = strawberry.field(description="The agent")


@strawberry.type
class AgentConnection:
    """Paginated connection of agents."""

    edges: List[AgentEdge] = strawberry.field(description="List of edges")
    page_info: PageInfo = strawberry.field(description="Pagination info")

    @strawberry.field(description="Get all agent nodes directly")
    def nodes(self) -> List[ProcessHeatAgentType]:
        """Get all agent nodes without edge wrapper."""
        return [edge.node for edge in self.edges]


# =============================================================================
# Statistics Types
# =============================================================================


@strawberry.type
class CategoryStatsType:
    """Statistics for an agent category."""

    category: str = strawberry.field(description="Category name")
    count: int = strawberry.field(description="Number of agents")
    available_count: int = strawberry.field(description="Available agents")
    total_invocations: int = strawberry.field(description="Total invocations")


@strawberry.type
class RegistryStatsType:
    """Overall registry statistics."""

    total_agents: int = strawberry.field(description="Total registered agents")
    by_category: List[CategoryStatsType] = strawberry.field(
        description="Stats by category"
    )
    by_type: JSON = strawberry.field(description="Agent counts by type")
    by_priority: JSON = strawberry.field(description="Agent counts by priority")
    by_complexity: JSON = strawberry.field(description="Agent counts by complexity")
    total_addressable_market_billions: float = strawberry.field(
        description="Total addressable market in billions USD"
    )
    loaded_instances: int = strawberry.field(description="Currently loaded instances")
