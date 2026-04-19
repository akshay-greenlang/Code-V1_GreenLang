"""
Agent Registry Service

This module provides agent lifecycle management, version control, and discovery
for the GreenLang Agent Factory.

The registry manages:
- Agent registration and metadata
- Lifecycle state machine (DRAFT -> CERTIFIED -> RETIRED)
- Version management with semantic versioning
- Capability-based discovery

Example:
    >>> service = AgentRegistryService(repository)
    >>> agent = await service.register_agent(agent_spec)
    >>> agents = await service.search_agents("carbon emissions")
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""

    DRAFT = "DRAFT"  # Initial state, under development
    EXPERIMENTAL = "EXPERIMENTAL"  # Testing/validation
    CERTIFIED = "CERTIFIED"  # Production ready
    DEPRECATED = "DEPRECATED"  # Marked for retirement
    RETIRED = "RETIRED"  # No longer available


# Valid state transitions
STATE_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.DRAFT: {AgentState.EXPERIMENTAL, AgentState.RETIRED},
    AgentState.EXPERIMENTAL: {AgentState.CERTIFIED, AgentState.DRAFT, AgentState.RETIRED},
    AgentState.CERTIFIED: {AgentState.DEPRECATED},
    AgentState.DEPRECATED: {AgentState.RETIRED, AgentState.CERTIFIED},
    AgentState.RETIRED: set(),  # Terminal state
}


class AgentSpec(BaseModel):
    """Agent specification from pack.yaml."""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Semantic version")
    description: Optional[str] = Field(None, description="Agent description")
    category: str = Field(..., description="Agent category (emissions, cbam, csrd, etc.)")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # Compute configuration
    entrypoint: str = Field(..., description="Python entrypoint")
    deterministic: bool = Field(True, description="Is agent deterministic (zero-hallucination)")

    # Input/Output schema
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Regulatory scope
    regulatory_frameworks: List[str] = Field(
        default_factory=list,
        description="Applicable frameworks (CBAM, CSRD, EUDR, etc.)"
    )

    # Metadata
    owners: List[str] = Field(default_factory=list)
    documentation_url: Optional[str] = None

    @validator("agent_id")
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent ID format."""
        if not v or not v.strip():
            raise ValueError("agent_id cannot be empty")
        # Format: category/name_version (e.g., emissions/carbon_calculator_v1)
        if "/" not in v:
            raise ValueError("agent_id must be in format: category/name")
        return v.strip().lower()

    @validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("version must be in format: X.Y.Z")
        for part in parts:
            if not part.isdigit():
                raise ValueError("version parts must be numeric")
        return v


class AgentRegistration(BaseModel):
    """Registered agent with metadata."""

    id: str = Field(..., description="Database ID")
    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Current version")
    state: AgentState = Field(AgentState.DRAFT, description="Lifecycle state")
    category: str = Field(..., description="Agent category")
    tags: List[str] = Field(default_factory=list)

    tenant_id: str = Field(..., description="Owner tenant")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Full spec reference
    spec: Optional[AgentSpec] = None

    # Metrics
    invocation_count: int = Field(0, description="Total invocations")
    success_rate: float = Field(1.0, description="Success rate (0-1)")


class AgentFilters(BaseModel):
    """Filters for agent queries."""

    category: Optional[str] = None
    state: Optional[AgentState] = None
    tags: Optional[List[str]] = None
    regulatory_frameworks: Optional[List[str]] = None
    search_query: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""

    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    sort_by: str = Field("created_at")
    sort_order: str = Field("desc")


class AgentRegistryService:
    """
    Agent Registry Service.

    This service manages agent lifecycle including:
    - Registration with validation
    - State machine transitions
    - Version management
    - Search and discovery

    Attributes:
        repository: Database repository for persistence
        capability_index: Index of agent capabilities for discovery

    Example:
        >>> service = AgentRegistryService(repository)
        >>> agent = await service.register_agent(spec)
        >>> await service.transition_state(agent.agent_id, AgentState.CERTIFIED)
    """

    def __init__(
        self,
        repository: Optional[Any] = None,
        cache: Optional[Any] = None,
    ):
        """
        Initialize the Agent Registry Service.

        Args:
            repository: Database repository
            cache: Optional cache layer (Redis)
        """
        self.repository = repository
        self.cache = cache

        # In-memory indexes (populated from DB on startup)
        self._agents: Dict[str, AgentRegistration] = {}
        self._capability_index: Dict[str, Set[str]] = {}
        self._category_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

        logger.info("AgentRegistryService initialized")

    async def register_agent(
        self,
        spec: AgentSpec,
        tenant_id: str,
    ) -> AgentRegistration:
        """
        Register a new agent.

        Args:
            spec: Agent specification
            tenant_id: Owning tenant

        Returns:
            Registered agent

        Raises:
            ValueError: If agent already exists
        """
        # Check for duplicate
        if spec.agent_id in self._agents:
            raise ValueError(f"Agent {spec.agent_id} already exists")

        # Create registration
        registration = AgentRegistration(
            id=f"agent-{len(self._agents) + 1:06d}",
            agent_id=spec.agent_id,
            name=spec.name,
            version=spec.version,
            state=AgentState.DRAFT,
            category=spec.category,
            tags=spec.tags,
            tenant_id=tenant_id,
            spec=spec,
        )

        # Store in repository
        if self.repository:
            await self.repository.create(registration)

        # Update indexes
        self._agents[spec.agent_id] = registration
        self._update_indexes(registration)

        # Invalidate cache
        if self.cache:
            await self.cache.delete(f"agent:{spec.agent_id}")

        logger.info(f"Registered agent {spec.agent_id} for tenant {tenant_id}")
        return registration

    async def get_agent(
        self,
        agent_id: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[AgentRegistration]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier
            tenant_id: Optional tenant filter for isolation

        Returns:
            Agent registration or None if not found
        """
        # Check cache first
        if self.cache:
            cached = await self.cache.get(f"agent:{agent_id}")
            if cached:
                return AgentRegistration.parse_raw(cached)

        # Check in-memory index
        agent = self._agents.get(agent_id)

        # Apply tenant isolation
        if agent and tenant_id and agent.tenant_id != tenant_id:
            return None

        # Cache result
        if agent and self.cache:
            await self.cache.set(
                f"agent:{agent_id}",
                agent.json(),
                ex=300  # 5 minute TTL
            )

        return agent

    async def list_agents(
        self,
        filters: Optional[AgentFilters] = None,
        pagination: Optional[PaginationParams] = None,
        tenant_id: Optional[str] = None,
    ) -> List[AgentRegistration]:
        """
        List agents with filtering and pagination.

        Args:
            filters: Optional filters
            pagination: Pagination parameters
            tenant_id: Tenant for isolation

        Returns:
            List of matching agents
        """
        filters = filters or AgentFilters()
        pagination = pagination or PaginationParams()

        # Start with all agents
        results = list(self._agents.values())

        # Apply tenant isolation
        if tenant_id:
            results = [a for a in results if a.tenant_id == tenant_id]

        # Apply filters
        if filters.category:
            results = [a for a in results if a.category == filters.category]

        if filters.state:
            results = [a for a in results if a.state == filters.state]

        if filters.tags:
            results = [
                a for a in results
                if any(tag in a.tags for tag in filters.tags)
            ]

        if filters.search_query:
            query = filters.search_query.lower()
            results = [
                a for a in results
                if query in a.name.lower() or
                   query in a.agent_id.lower() or
                   any(query in tag.lower() for tag in a.tags)
            ]

        # Sort
        reverse = pagination.sort_order == "desc"
        results.sort(
            key=lambda a: getattr(a, pagination.sort_by, a.created_at),
            reverse=reverse
        )

        # Paginate
        start = pagination.offset
        end = start + pagination.limit
        return results[start:end]

    async def update_agent(
        self,
        agent_id: str,
        updates: Dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> AgentRegistration:
        """
        Update agent metadata.

        Args:
            agent_id: Agent to update
            updates: Fields to update
            tenant_id: Tenant for isolation

        Returns:
            Updated agent

        Raises:
            ValueError: If agent not found or not authorized
        """
        agent = await self.get_agent(agent_id, tenant_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Apply updates
        for key, value in updates.items():
            if hasattr(agent, key) and key not in ("id", "agent_id", "created_at"):
                setattr(agent, key, value)

        agent.updated_at = datetime.utcnow()

        # Update repository
        if self.repository:
            await self.repository.update(agent.id, agent)

        # Update cache
        if self.cache:
            await self.cache.set(f"agent:{agent_id}", agent.json(), ex=300)

        # Update indexes
        self._update_indexes(agent)

        logger.info(f"Updated agent {agent_id}")
        return agent

    async def delete_agent(
        self,
        agent_id: str,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Soft delete an agent.

        Args:
            agent_id: Agent to delete
            tenant_id: Tenant for isolation

        Returns:
            True if deleted

        Raises:
            ValueError: If agent is CERTIFIED (cannot delete certified agents)
        """
        agent = await self.get_agent(agent_id, tenant_id)
        if not agent:
            return False

        if agent.state == AgentState.CERTIFIED:
            raise ValueError("Cannot delete CERTIFIED agents. Deprecate first.")

        # Transition to RETIRED
        agent.state = AgentState.RETIRED
        agent.updated_at = datetime.utcnow()

        # Update repository
        if self.repository:
            await self.repository.update(agent.id, agent)

        # Remove from indexes
        self._remove_from_indexes(agent)
        del self._agents[agent_id]

        # Invalidate cache
        if self.cache:
            await self.cache.delete(f"agent:{agent_id}")

        logger.info(f"Deleted agent {agent_id}")
        return True

    async def transition_state(
        self,
        agent_id: str,
        target_state: AgentState,
        actor: str,
        tenant_id: Optional[str] = None,
    ) -> AgentRegistration:
        """
        Transition agent to a new state.

        Args:
            agent_id: Agent to transition
            target_state: Target state
            actor: User making the transition
            tenant_id: Tenant for isolation

        Returns:
            Updated agent

        Raises:
            ValueError: If transition is not allowed
        """
        agent = await self.get_agent(agent_id, tenant_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Validate transition
        allowed = STATE_TRANSITIONS.get(agent.state, set())
        if target_state not in allowed:
            raise ValueError(
                f"Cannot transition from {agent.state} to {target_state}. "
                f"Allowed: {allowed}"
            )

        old_state = agent.state
        agent.state = target_state
        agent.updated_at = datetime.utcnow()

        # Update repository
        if self.repository:
            await self.repository.update(agent.id, agent)

        # Update cache
        if self.cache:
            await self.cache.set(f"agent:{agent_id}", agent.json(), ex=300)

        logger.info(
            f"Agent {agent_id} transitioned from {old_state} to {target_state} by {actor}"
        )
        return agent

    async def search_agents(
        self,
        query: str,
        filters: Optional[AgentFilters] = None,
        limit: int = 20,
        tenant_id: Optional[str] = None,
    ) -> List[AgentRegistration]:
        """
        Search agents by text query.

        Args:
            query: Search query
            filters: Additional filters
            limit: Max results
            tenant_id: Tenant for isolation

        Returns:
            Matching agents ordered by relevance
        """
        filters = filters or AgentFilters()
        filters.search_query = query

        return await self.list_agents(
            filters=filters,
            pagination=PaginationParams(limit=limit),
            tenant_id=tenant_id,
        )

    async def get_agents_by_capability(
        self,
        capability: str,
        tenant_id: Optional[str] = None,
    ) -> List[AgentRegistration]:
        """
        Get agents that provide a specific capability.

        Args:
            capability: Capability to search for
            tenant_id: Tenant for isolation

        Returns:
            Agents with the capability
        """
        agent_ids = self._capability_index.get(capability, set())
        results = []

        for agent_id in agent_ids:
            agent = await self.get_agent(agent_id, tenant_id)
            if agent:
                results.append(agent)

        return results

    async def get_agents_by_category(
        self,
        category: str,
        tenant_id: Optional[str] = None,
    ) -> List[AgentRegistration]:
        """
        Get all agents in a category.

        Args:
            category: Category name
            tenant_id: Tenant for isolation

        Returns:
            Agents in the category
        """
        return await self.list_agents(
            filters=AgentFilters(category=category),
            tenant_id=tenant_id,
        )

    def _update_indexes(self, agent: AgentRegistration) -> None:
        """Update search indexes for an agent."""
        # Category index
        if agent.category not in self._category_index:
            self._category_index[agent.category] = set()
        self._category_index[agent.category].add(agent.agent_id)

        # Tag index
        for tag in agent.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(agent.agent_id)

    def _remove_from_indexes(self, agent: AgentRegistration) -> None:
        """Remove agent from search indexes."""
        # Category index
        if agent.category in self._category_index:
            self._category_index[agent.category].discard(agent.agent_id)

        # Tag index
        for tag in agent.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(agent.agent_id)

    async def get_statistics(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get registry statistics."""
        agents = list(self._agents.values())

        if tenant_id:
            agents = [a for a in agents if a.tenant_id == tenant_id]

        return {
            "total_agents": len(agents),
            "by_state": {
                state.value: len([a for a in agents if a.state == state])
                for state in AgentState
            },
            "by_category": {
                cat: len(ids)
                for cat, ids in self._category_index.items()
            },
            "total_invocations": sum(a.invocation_count for a in agents),
        }
