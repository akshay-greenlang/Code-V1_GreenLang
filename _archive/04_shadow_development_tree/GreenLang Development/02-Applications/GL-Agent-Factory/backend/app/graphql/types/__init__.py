"""
GreenLang GraphQL Types Package

This package contains all GraphQL type definitions for the Process Heat agents API.

Types:
- Agent types: Core agent definitions and metadata
- Calculation types: Calculation results with provenance
- Event types: Real-time event notifications

Usage:
    from app.graphql.types import (
        ProcessHeatAgentType,
        CalculationResultType,
        AgentEventType,
    )
"""

from app.graphql.types.agent import (
    ProcessHeatAgentType,
    AgentInfoType,
    AgentStatusEnum,
    AgentCategoryEnum,
    AgentTypeEnum,
    AgentPriorityEnum,
    AgentComplexityEnum,
    HealthStatusType,
    AgentMetricsType,
    AgentConfigType,
    AgentConnection,
    AgentEdge,
)

from app.graphql.types.calculation import (
    CalculationResultType,
    CalculationStatusEnum,
    CalculationInputType,
    CalculationOutputType,
    EmissionFactorType,
    ProvenanceType,
    UnitConversionType,
    ValidationResultType,
    QualityScoreType,
)

from app.graphql.types.events import (
    AgentEventType,
    EventTypeEnum,
    ProgressType,
    ExecutionEventType,
    SystemEventType,
    ComplianceEventType,
    CalculationProgressType,
)

__all__ = [
    # Agent types
    "ProcessHeatAgentType",
    "AgentInfoType",
    "AgentStatusEnum",
    "AgentCategoryEnum",
    "AgentTypeEnum",
    "AgentPriorityEnum",
    "AgentComplexityEnum",
    "HealthStatusType",
    "AgentMetricsType",
    "AgentConfigType",
    "AgentConnection",
    "AgentEdge",
    # Calculation types
    "CalculationResultType",
    "CalculationStatusEnum",
    "CalculationInputType",
    "CalculationOutputType",
    "EmissionFactorType",
    "ProvenanceType",
    "UnitConversionType",
    "ValidationResultType",
    "QualityScoreType",
    # Event types
    "AgentEventType",
    "EventTypeEnum",
    "ProgressType",
    "ExecutionEventType",
    "SystemEventType",
    "ComplianceEventType",
    "CalculationProgressType",
]
