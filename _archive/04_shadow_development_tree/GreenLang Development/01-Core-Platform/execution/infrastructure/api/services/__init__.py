"""
GreenLang GraphQL Service Layer

This package provides service layer implementations for GraphQL resolvers,
abstracting business logic from the GraphQL schema definitions.

Services:
    - AgentService: Agent management and monitoring
    - CalculationService: Calculation job orchestration
    - ComplianceService: Compliance report generation

Example:
    >>> from greenlang.infrastructure.api.services import (
    ...     AgentService,
    ...     CalculationService,
    ...     ComplianceService
    ... )
    >>> agent_service = AgentService()
    >>> agents = await agent_service.get_all_agents()
"""

from greenlang.infrastructure.api.services.agent_service import AgentService
from greenlang.infrastructure.api.services.calculation_service import CalculationService
from greenlang.infrastructure.api.services.compliance_service import ComplianceService

__all__ = [
    "AgentService",
    "CalculationService",
    "ComplianceService",
]
