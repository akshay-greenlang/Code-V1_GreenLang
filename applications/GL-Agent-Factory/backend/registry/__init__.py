"""
GreenLang Agent Registry

This package provides the centralized registry infrastructure for managing
AI agents with PostgreSQL backend, FastAPI endpoints, and version management.

Components:
    - models: Pydantic models for request/response validation
    - db_models: SQLAlchemy ORM models for database operations
    - service: AgentRegistryService for business logic
    - api: FastAPI endpoints for CRUD operations
    - cli: CLI integration for push/pull/search commands
    - migrations: Alembic database migrations

Example:
    >>> from backend.registry import AgentRegistryService, AgentRecord
    >>> service = AgentRegistryService(session)
    >>> agent = await service.create_agent(agent_data)
"""

from backend.registry.models import (
    AgentRecord,
    AgentVersion,
    AgentStatus,
    CertificationStatus,
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentResponse,
    AgentListResponse,
    AgentSearchRequest,
    VersionCreateRequest,
    VersionResponse,
    PublishRequest,
)
from backend.registry.service import AgentRegistryService

__all__ = [
    # Models
    "AgentRecord",
    "AgentVersion",
    "AgentStatus",
    "CertificationStatus",
    "AgentCreateRequest",
    "AgentUpdateRequest",
    "AgentResponse",
    "AgentListResponse",
    "AgentSearchRequest",
    "VersionCreateRequest",
    "VersionResponse",
    "PublishRequest",
    # Service
    "AgentRegistryService",
]

__version__ = "1.0.0"
