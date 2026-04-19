"""
Agent Registry Service

This module provides the AgentRegistryService class implementing all business
logic for the Agent Registry including:
- Agent CRUD operations
- Version management
- Search functionality
- Publishing workflow
- Statistics and analytics

The service follows GreenLang's zero-hallucination principle with deterministic
operations and comprehensive provenance tracking.

Example:
    >>> from backend.registry.service import AgentRegistryService
    >>> from sqlalchemy.ext.asyncio import AsyncSession
    >>>
    >>> async with get_session() as session:
    ...     service = AgentRegistryService(session)
    ...     agent = await service.create_agent(
    ...         name="carbon-calculator",
    ...         version="1.0.0",
    ...         category="emissions",
    ...         author="greenlang"
    ...     )
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.registry.db_models import AgentRecordDB, AgentVersionDB
from backend.registry.models import (
    AgentRecord,
    AgentVersion,
    AgentStatus,
    CertificationStatus,
)

logger = logging.getLogger(__name__)


class AgentRegistryService:
    """
    Agent Registry Service.

    Provides comprehensive agent lifecycle management including:
    - Registration with validation
    - CRUD operations with tenant isolation
    - Version management with semantic versioning
    - Full-text search and filtering
    - Publishing workflow with certifications
    - Statistics and analytics

    Attributes:
        session: SQLAlchemy async session
        _cache: Optional in-memory cache for frequently accessed data

    Example:
        >>> service = AgentRegistryService(session)
        >>> agent = await service.create_agent(
        ...     name="test-agent",
        ...     version="1.0.0",
        ...     category="test",
        ...     author="test-user"
        ... )
        >>> await service.publish_agent(agent.id, "1.0.0")
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        """
        Initialize the Agent Registry Service.

        Args:
            session: SQLAlchemy async session for database operations.
                    If None, operations will use in-memory storage.
        """
        self.session = session
        self._cache: Dict[str, Any] = {}
        self._in_memory_agents: Dict[UUID, AgentRecordDB] = {}
        self._in_memory_versions: Dict[UUID, AgentVersionDB] = {}
        logger.info("AgentRegistryService initialized")

    # =========================================================================
    # Agent CRUD Operations
    # =========================================================================

    async def create_agent(
        self,
        name: str,
        version: str,
        category: str,
        author: str,
        description: str = "",
        pack_yaml: Optional[Dict[str, Any]] = None,
        generated_code: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        documentation_url: Optional[str] = None,
        repository_url: Optional[str] = None,
        license: str = "Apache-2.0",
        tenant_id: Optional[str] = None,
    ) -> AgentRecordDB:
        """
        Create a new agent in the registry.

        Args:
            name: Unique agent name (will be lowercased)
            version: Initial semantic version (X.Y.Z)
            category: Agent category
            author: Agent author or owner
            description: Agent description
            pack_yaml: Pack configuration as JSON
            generated_code: Generated code artifacts as JSON
            tags: Searchable tags
            regulatory_frameworks: Applicable regulatory frameworks
            documentation_url: Documentation URL
            repository_url: Source repository URL
            license: License identifier
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Created agent record

        Raises:
            ValueError: If agent name already exists or validation fails
        """
        start_time = datetime.utcnow()
        logger.info(f"Creating agent: {name}")

        # Normalize name
        name = name.lower().strip()

        # Check for existing agent
        existing = await self.get_agent_by_name(name)
        if existing:
            raise ValueError(f"Agent with name '{name}' already exists")

        # Compute checksum
        checksum = self._compute_checksum(
            name=name,
            version=version,
            pack_yaml=pack_yaml or {},
            generated_code=generated_code or {},
        )

        # Create agent record
        agent = AgentRecordDB(
            id=uuid4(),
            name=name,
            version=version,
            description=description,
            category=category.lower().strip(),
            pack_yaml=pack_yaml or {},
            generated_code=generated_code or {},
            checksum=checksum,
            status="draft",
            author=author,
            tenant_id=UUID(tenant_id) if tenant_id else None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            downloads=0,
            certification_status=[],
            tags=tags or [],
            regulatory_frameworks=regulatory_frameworks or [],
            documentation_url=documentation_url,
            repository_url=repository_url,
            license=license,
        )

        # Persist to database
        if self.session:
            self.session.add(agent)
            await self.session.flush()
            await self.session.refresh(agent)

            # Create initial version
            await self._create_initial_version(agent)
        else:
            # In-memory storage for testing
            self._in_memory_agents[agent.id] = agent
            await self._create_initial_version(agent)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Created agent {name} (id={agent.id}) in {processing_time:.2f}ms")

        return agent

    async def get_agent(
        self,
        agent_id: UUID,
        tenant_id: Optional[str] = None,
    ) -> Optional[AgentRecordDB]:
        """
        Get agent by UUID.

        Args:
            agent_id: Agent UUID
            tenant_id: Optional tenant filter for isolation

        Returns:
            Agent record or None if not found
        """
        logger.debug(f"Getting agent: {agent_id}")

        if self.session:
            query = select(AgentRecordDB).where(AgentRecordDB.id == agent_id)

            if tenant_id:
                query = query.where(
                    or_(
                        AgentRecordDB.tenant_id == UUID(tenant_id),
                        AgentRecordDB.tenant_id.is_(None),
                    )
                )

            query = query.options(selectinload(AgentRecordDB.versions))
            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        else:
            agent = self._in_memory_agents.get(agent_id)
            if agent and tenant_id:
                if agent.tenant_id and str(agent.tenant_id) != tenant_id:
                    return None
            return agent

    async def get_agent_by_name(
        self,
        name: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[AgentRecordDB]:
        """
        Get agent by name.

        Args:
            name: Agent name (case-insensitive)
            tenant_id: Optional tenant filter

        Returns:
            Agent record or None if not found
        """
        name = name.lower().strip()

        if self.session:
            query = select(AgentRecordDB).where(AgentRecordDB.name == name)

            if tenant_id:
                query = query.where(
                    or_(
                        AgentRecordDB.tenant_id == UUID(tenant_id),
                        AgentRecordDB.tenant_id.is_(None),
                    )
                )

            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        else:
            for agent in self._in_memory_agents.values():
                if agent.name == name:
                    if tenant_id and agent.tenant_id:
                        if str(agent.tenant_id) != tenant_id:
                            continue
                    return agent
            return None

    async def list_agents(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        author: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Tuple[List[AgentRecordDB], int]:
        """
        List agents with filtering and pagination.

        Args:
            category: Filter by category
            status: Filter by status
            tags: Filter by tags (any match)
            regulatory_frameworks: Filter by frameworks (any match)
            author: Filter by author
            tenant_id: Tenant for isolation
            limit: Maximum results
            offset: Pagination offset
            sort_by: Sort field
            sort_order: Sort direction (asc/desc)

        Returns:
            Tuple of (agents list, total count)
        """
        logger.debug(f"Listing agents: category={category}, status={status}")

        if self.session:
            # Build base query
            query = select(AgentRecordDB)
            count_query = select(func.count(AgentRecordDB.id))

            # Apply filters
            filters = []

            if category:
                filters.append(AgentRecordDB.category == category.lower())
            if status:
                filters.append(AgentRecordDB.status == status.lower())
            if author:
                filters.append(AgentRecordDB.author == author)
            if tenant_id:
                filters.append(
                    or_(
                        AgentRecordDB.tenant_id == UUID(tenant_id),
                        AgentRecordDB.tenant_id.is_(None),
                    )
                )
            if tags:
                # PostgreSQL array overlap
                filters.append(AgentRecordDB.tags.overlap(tags))
            if regulatory_frameworks:
                filters.append(
                    AgentRecordDB.regulatory_frameworks.overlap(regulatory_frameworks)
                )

            if filters:
                query = query.where(and_(*filters))
                count_query = count_query.where(and_(*filters))

            # Get total count
            count_result = await self.session.execute(count_query)
            total = count_result.scalar() or 0

            # Apply sorting
            sort_column = getattr(AgentRecordDB, sort_by, AgentRecordDB.created_at)
            if sort_order.lower() == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())

            # Apply pagination
            query = query.offset(offset).limit(limit)

            # Load with versions
            query = query.options(selectinload(AgentRecordDB.versions))

            result = await self.session.execute(query)
            agents = list(result.scalars().all())

            return agents, total
        else:
            # In-memory filtering
            agents = list(self._in_memory_agents.values())

            if category:
                agents = [a for a in agents if a.category == category.lower()]
            if status:
                agents = [a for a in agents if a.status == status.lower()]
            if author:
                agents = [a for a in agents if a.author == author]
            if tags:
                agents = [
                    a for a in agents
                    if any(t in (a.tags or []) for t in tags)
                ]

            total = len(agents)

            # Sort
            reverse = sort_order.lower() == "desc"
            agents.sort(
                key=lambda a: getattr(a, sort_by, a.created_at) or datetime.min,
                reverse=reverse,
            )

            # Paginate
            agents = agents[offset:offset + limit]

            return agents, total

    async def update_agent(
        self,
        agent_id: UUID,
        updates: Dict[str, Any],
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[AgentRecordDB]:
        """
        Update agent metadata.

        Args:
            agent_id: Agent UUID
            updates: Fields to update
            tenant_id: Tenant for isolation
            user_id: User performing update

        Returns:
            Updated agent or None if not found

        Raises:
            PermissionError: If user not authorized
        """
        logger.info(f"Updating agent {agent_id}: {list(updates.keys())}")

        agent = await self.get_agent(agent_id, tenant_id)
        if not agent:
            return None

        # Check authorization
        if tenant_id and agent.tenant_id:
            if str(agent.tenant_id) != tenant_id:
                raise PermissionError("Not authorized to update this agent")

        # Allowed update fields
        allowed_fields = {
            "description",
            "pack_yaml",
            "generated_code",
            "tags",
            "regulatory_frameworks",
            "documentation_url",
            "repository_url",
        }

        # Apply updates
        for key, value in updates.items():
            if key in allowed_fields:
                setattr(agent, key, value)

        agent.updated_at = datetime.utcnow()

        # Recompute checksum if content changed
        if "pack_yaml" in updates or "generated_code" in updates:
            agent.checksum = self._compute_checksum(
                name=agent.name,
                version=agent.version,
                pack_yaml=agent.pack_yaml,
                generated_code=agent.generated_code,
            )

        if self.session:
            await self.session.flush()
            await self.session.refresh(agent)

        return agent

    async def delete_agent(
        self,
        agent_id: UUID,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Delete (or deprecate) an agent.

        Published agents are deprecated instead of deleted.
        Draft agents are permanently removed.

        Args:
            agent_id: Agent UUID
            tenant_id: Tenant for isolation
            user_id: User performing deletion

        Returns:
            True if agent was deleted/deprecated

        Raises:
            PermissionError: If user not authorized
        """
        logger.info(f"Deleting agent: {agent_id}")

        agent = await self.get_agent(agent_id, tenant_id)
        if not agent:
            return False

        # Check authorization
        if tenant_id and agent.tenant_id:
            if str(agent.tenant_id) != tenant_id:
                raise PermissionError("Not authorized to delete this agent")

        if agent.status == "published":
            # Deprecate instead of delete
            agent.status = "deprecated"
            agent.updated_at = datetime.utcnow()
            if self.session:
                await self.session.flush()
        else:
            # Actually delete draft agents
            if self.session:
                await self.session.delete(agent)
                await self.session.flush()
            else:
                del self._in_memory_agents[agent_id]

        return True

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search_agents(
        self,
        query: str,
        category: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[AgentRecordDB], int]:
        """
        Search agents by text query.

        Uses PostgreSQL full-text search for efficient searching
        across name, description, and tags.

        Args:
            query: Search query string
            category: Optional category filter
            status: Optional status filter
            tags: Optional tag filter
            tenant_id: Tenant for isolation
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (matching agents, total count)
        """
        logger.info(f"Searching agents: query='{query}'")

        if self.session:
            # PostgreSQL full-text search
            search_query = select(AgentRecordDB).where(
                or_(
                    AgentRecordDB.name.ilike(f"%{query}%"),
                    AgentRecordDB.description.ilike(f"%{query}%"),
                    AgentRecordDB.tags.any(query.lower()),
                )
            )

            count_query = select(func.count(AgentRecordDB.id)).where(
                or_(
                    AgentRecordDB.name.ilike(f"%{query}%"),
                    AgentRecordDB.description.ilike(f"%{query}%"),
                    AgentRecordDB.tags.any(query.lower()),
                )
            )

            # Additional filters
            if category:
                search_query = search_query.where(AgentRecordDB.category == category.lower())
                count_query = count_query.where(AgentRecordDB.category == category.lower())
            if status:
                search_query = search_query.where(AgentRecordDB.status == status.lower())
                count_query = count_query.where(AgentRecordDB.status == status.lower())
            if tenant_id:
                search_query = search_query.where(
                    or_(
                        AgentRecordDB.tenant_id == UUID(tenant_id),
                        AgentRecordDB.tenant_id.is_(None),
                    )
                )
                count_query = count_query.where(
                    or_(
                        AgentRecordDB.tenant_id == UUID(tenant_id),
                        AgentRecordDB.tenant_id.is_(None),
                    )
                )

            # Get total count
            count_result = await self.session.execute(count_query)
            total = count_result.scalar() or 0

            # Order by downloads (relevance proxy)
            search_query = search_query.order_by(AgentRecordDB.downloads.desc())

            # Pagination
            search_query = search_query.offset(offset).limit(limit)

            result = await self.session.execute(search_query)
            agents = list(result.scalars().all())

            return agents, total
        else:
            # In-memory search
            query_lower = query.lower()
            agents = [
                a for a in self._in_memory_agents.values()
                if (
                    query_lower in a.name.lower()
                    or query_lower in (a.description or "").lower()
                    or any(query_lower in t.lower() for t in (a.tags or []))
                )
            ]

            if category:
                agents = [a for a in agents if a.category == category.lower()]
            if status:
                agents = [a for a in agents if a.status == status.lower()]

            total = len(agents)
            agents = agents[offset:offset + limit]

            return agents, total

    # =========================================================================
    # Version Management
    # =========================================================================

    async def create_version(
        self,
        agent_id: UUID,
        version: str,
        changelog: str = "",
        breaking_changes: bool = False,
        release_notes: str = "",
        pack_yaml: Optional[Dict[str, Any]] = None,
        generated_code: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> AgentVersionDB:
        """
        Create a new version for an agent.

        Args:
            agent_id: Agent UUID
            version: Semantic version string
            changelog: Version changelog
            breaking_changes: Whether version has breaking changes
            release_notes: Detailed release notes
            pack_yaml: Updated pack configuration
            generated_code: Updated generated code
            user_id: User creating version

        Returns:
            Created version record

        Raises:
            ValueError: If agent not found or version already exists
        """
        logger.info(f"Creating version {version} for agent {agent_id}")

        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Check version doesn't exist
        existing = await self.get_version(agent_id, version)
        if existing:
            raise ValueError(f"Version {version} already exists for agent {agent_id}")

        # Validate version is higher than current
        if not self._is_higher_version(version, agent.version):
            raise ValueError(
                f"New version {version} must be higher than current version {agent.version}"
            )

        # Compute checksum for version artifacts
        checksum = self._compute_checksum(
            name=agent.name,
            version=version,
            pack_yaml=pack_yaml or agent.pack_yaml,
            generated_code=generated_code or agent.generated_code,
        )

        # Unset is_latest on previous versions
        await self._unset_latest_version(agent_id)

        # Create version record
        ver = AgentVersionDB(
            id=uuid4(),
            agent_id=agent_id,
            version=version,
            changelog=changelog,
            breaking_changes=breaking_changes,
            release_notes=release_notes,
            checksum=checksum,
            is_latest=True,
            created_at=datetime.utcnow(),
            downloads=0,
            pack_yaml_snapshot=pack_yaml or agent.pack_yaml,
            generated_code_snapshot=generated_code or agent.generated_code,
        )

        # Update agent's current version
        agent.version = version
        agent.updated_at = datetime.utcnow()

        if pack_yaml:
            agent.pack_yaml = pack_yaml
        if generated_code:
            agent.generated_code = generated_code

        agent.checksum = checksum

        if self.session:
            self.session.add(ver)
            await self.session.flush()
            await self.session.refresh(ver)
        else:
            self._in_memory_versions[ver.id] = ver

        return ver

    async def list_versions(
        self,
        agent_id: UUID,
        include_deprecated: bool = False,
    ) -> List[AgentVersionDB]:
        """
        List all versions of an agent.

        Args:
            agent_id: Agent UUID
            include_deprecated: Include deprecated versions

        Returns:
            List of version records, newest first
        """
        if self.session:
            query = select(AgentVersionDB).where(AgentVersionDB.agent_id == agent_id)

            if not include_deprecated:
                query = query.where(AgentVersionDB.deprecated_at.is_(None))

            query = query.order_by(AgentVersionDB.created_at.desc())

            result = await self.session.execute(query)
            return list(result.scalars().all())
        else:
            versions = [
                v for v in self._in_memory_versions.values()
                if v.agent_id == agent_id
            ]
            if not include_deprecated:
                versions = [v for v in versions if v.deprecated_at is None]
            versions.sort(key=lambda v: v.created_at, reverse=True)
            return versions

    async def get_version(
        self,
        agent_id: UUID,
        version: str,
    ) -> Optional[AgentVersionDB]:
        """
        Get a specific version of an agent.

        Args:
            agent_id: Agent UUID
            version: Version string or 'latest'

        Returns:
            Version record or None if not found
        """
        if self.session:
            query = select(AgentVersionDB).where(AgentVersionDB.agent_id == agent_id)

            if version.lower() == "latest":
                query = query.where(AgentVersionDB.is_latest == True)
            else:
                query = query.where(AgentVersionDB.version == version)

            result = await self.session.execute(query)
            return result.scalar_one_or_none()
        else:
            for v in self._in_memory_versions.values():
                if v.agent_id == agent_id:
                    if version.lower() == "latest" and v.is_latest:
                        return v
                    if v.version == version:
                        return v
            return None

    async def _create_initial_version(self, agent: AgentRecordDB) -> AgentVersionDB:
        """Create initial version for a new agent."""
        ver = AgentVersionDB(
            id=uuid4(),
            agent_id=agent.id,
            version=agent.version,
            changelog="Initial version",
            breaking_changes=False,
            release_notes="",
            checksum=agent.checksum,
            is_latest=True,
            created_at=datetime.utcnow(),
            downloads=0,
            pack_yaml_snapshot=agent.pack_yaml,
            generated_code_snapshot=agent.generated_code,
        )

        if self.session:
            self.session.add(ver)
            await self.session.flush()
        else:
            self._in_memory_versions[ver.id] = ver

        return ver

    async def _unset_latest_version(self, agent_id: UUID) -> None:
        """Unset is_latest flag on all versions of an agent."""
        if self.session:
            await self.session.execute(
                update(AgentVersionDB)
                .where(AgentVersionDB.agent_id == agent_id)
                .values(is_latest=False)
            )
        else:
            for v in self._in_memory_versions.values():
                if v.agent_id == agent_id:
                    v.is_latest = False

    # =========================================================================
    # Publishing Workflow
    # =========================================================================

    async def publish_agent(
        self,
        agent_id: UUID,
        version: str,
        release_notes: Optional[str] = None,
        certifications: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> AgentVersionDB:
        """
        Publish an agent version.

        Args:
            agent_id: Agent UUID
            version: Version to publish
            release_notes: Optional release notes
            certifications: Frameworks to certify for
            user_id: User publishing

        Returns:
            Published version record

        Raises:
            ValueError: If agent/version not found or already published
        """
        logger.info(f"Publishing agent {agent_id} version {version}")

        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        ver = await self.get_version(agent_id, version)
        if not ver:
            raise ValueError(f"Version {version} not found for agent {agent_id}")

        if ver.published_at:
            raise ValueError(f"Version {version} is already published")

        # Update version
        ver.published_at = datetime.utcnow()
        if release_notes:
            ver.release_notes = release_notes

        # Generate artifact path (placeholder)
        ver.artifact_path = f"s3://greenlang-registry/{agent.name}/{version}/agent.tar.gz"

        # Update agent status
        agent.status = "published"
        agent.updated_at = datetime.utcnow()

        # Add certifications
        if certifications:
            cert_status = agent.certification_status or []
            for framework in certifications:
                cert_status.append({
                    "framework": framework,
                    "certified": True,
                    "certified_at": datetime.utcnow().isoformat(),
                    "certified_by": user_id,
                })
            agent.certification_status = cert_status

        if self.session:
            await self.session.flush()
            await self.session.refresh(ver)

        return ver

    async def deprecate_agent(
        self,
        agent_id: UUID,
        user_id: Optional[str] = None,
    ) -> Optional[AgentRecordDB]:
        """
        Deprecate an agent.

        Args:
            agent_id: Agent UUID
            user_id: User deprecating

        Returns:
            Deprecated agent or None if not found
        """
        logger.info(f"Deprecating agent: {agent_id}")

        agent = await self.get_agent(agent_id)
        if not agent:
            return None

        agent.status = "deprecated"
        agent.updated_at = datetime.utcnow()

        if self.session:
            await self.session.flush()
            await self.session.refresh(agent)

        return agent

    # =========================================================================
    # Download Management
    # =========================================================================

    async def get_download(
        self,
        agent_id: UUID,
        version: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get download information for an agent.

        Args:
            agent_id: Agent UUID
            version: Optional specific version
            user_id: User downloading

        Returns:
            Download info dict or None if not available
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return None

        if agent.status != "published":
            return None

        ver = await self.get_version(agent_id, version or "latest")
        if not ver or not ver.published_at:
            return None

        return {
            "version": ver.version,
            "artifact_path": ver.artifact_path,
            "checksum": ver.checksum,
            "size_bytes": None,
        }

    async def increment_download(
        self,
        agent_id: UUID,
        version: str,
    ) -> None:
        """
        Increment download counter for an agent/version.

        Args:
            agent_id: Agent UUID
            version: Version downloaded
        """
        if self.session:
            # Increment agent downloads
            await self.session.execute(
                update(AgentRecordDB)
                .where(AgentRecordDB.id == agent_id)
                .values(downloads=AgentRecordDB.downloads + 1)
            )

            # Increment version downloads
            await self.session.execute(
                update(AgentVersionDB)
                .where(
                    and_(
                        AgentVersionDB.agent_id == agent_id,
                        AgentVersionDB.version == version,
                    )
                )
                .values(downloads=AgentVersionDB.downloads + 1)
            )

            await self.session.flush()
        else:
            agent = self._in_memory_agents.get(agent_id)
            if agent:
                agent.downloads += 1
            for v in self._in_memory_versions.values():
                if v.agent_id == agent_id and v.version == version:
                    v.downloads += 1
                    break

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_statistics(
        self,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get registry statistics.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Statistics dictionary
        """
        if self.session:
            # Total agents
            agents_query = select(func.count(AgentRecordDB.id))
            if tenant_id:
                agents_query = agents_query.where(
                    or_(
                        AgentRecordDB.tenant_id == UUID(tenant_id),
                        AgentRecordDB.tenant_id.is_(None),
                    )
                )
            agents_result = await self.session.execute(agents_query)
            total_agents = agents_result.scalar() or 0

            # Total versions
            versions_result = await self.session.execute(
                select(func.count(AgentVersionDB.id))
            )
            total_versions = versions_result.scalar() or 0

            # Total downloads
            downloads_result = await self.session.execute(
                select(func.sum(AgentRecordDB.downloads))
            )
            total_downloads = downloads_result.scalar() or 0

            # By status
            status_query = select(
                AgentRecordDB.status,
                func.count(AgentRecordDB.id),
            ).group_by(AgentRecordDB.status)
            status_result = await self.session.execute(status_query)
            by_status = {row[0]: row[1] for row in status_result.all()}

            # By category
            category_query = select(
                AgentRecordDB.category,
                func.count(AgentRecordDB.id),
            ).group_by(AgentRecordDB.category)
            category_result = await self.session.execute(category_query)
            by_category = {row[0]: row[1] for row in category_result.all()}

            return {
                "total_agents": total_agents,
                "total_versions": total_versions,
                "total_downloads": total_downloads,
                "by_status": by_status,
                "by_category": by_category,
                "recent_activity": [],
            }
        else:
            agents = list(self._in_memory_agents.values())
            return {
                "total_agents": len(agents),
                "total_versions": len(self._in_memory_versions),
                "total_downloads": sum(a.downloads for a in agents),
                "by_status": {},
                "by_category": {},
                "recent_activity": [],
            }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_checksum(
        self,
        name: str,
        version: str,
        pack_yaml: Dict[str, Any],
        generated_code: Dict[str, Any],
    ) -> str:
        """
        Compute SHA-256 checksum for agent content.

        Args:
            name: Agent name
            version: Version string
            pack_yaml: Pack configuration
            generated_code: Generated code

        Returns:
            Checksum string with 'sha256:' prefix
        """
        content = f"{name}{version}{str(pack_yaml)}{str(generated_code)}"
        hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"sha256:{hash_value}"

    def _is_higher_version(self, new_version: str, current_version: str) -> bool:
        """
        Check if new version is higher than current.

        Args:
            new_version: New version string
            current_version: Current version string

        Returns:
            True if new version is higher
        """
        try:
            new_parts = [int(x) for x in new_version.split("-")[0].split(".")]
            current_parts = [int(x) for x in current_version.split("-")[0].split(".")]

            for n, c in zip(new_parts, current_parts):
                if n > c:
                    return True
                if n < c:
                    return False
            return False
        except (ValueError, IndexError):
            return True  # Allow if parsing fails

    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """
        Parse semantic version string.

        Args:
            version: Version string (X.Y.Z)

        Returns:
            Tuple of (major, minor, patch)
        """
        parts = version.split("-")[0].split(".")
        return (
            int(parts[0]) if len(parts) > 0 else 0,
            int(parts[1]) if len(parts) > 1 else 0,
            int(parts[2]) if len(parts) > 2 else 0,
        )
