"""
GreenLang Process Heat Agents - Shared Libraries

This module provides shared utilities, base classes, and calculation libraries
for all process heat agents. These components ensure consistency, provenance
tracking, and regulatory compliance across the agent ecosystem.

Components:
    - BaseProcessHeatAgent: Enhanced base class with AI/ML, safety, and audit
    - ThermalIQCalculationLibrary: GL-009 THERMALIQ calculation engine
    - MultiAgentCoordinator: Agent coordination and orchestration utilities
    - ProvenanceTracker: SHA-256 provenance tracking for audit trails
    - AuditLogger: Comprehensive audit trail management

Example:
    >>> from greenlang.agents.process_heat.shared import BaseProcessHeatAgent
    >>> from greenlang.agents.process_heat.shared import ThermalIQCalculationLibrary
    >>>
    >>> class MyAgent(BaseProcessHeatAgent):
    ...     def process(self, input_data):
    ...         calc = ThermalIQCalculationLibrary()
    ...         return calc.calculate_boiler_efficiency(input_data)
"""

from greenlang.agents.process_heat.shared.base_agent import (
    BaseProcessHeatAgent,
    AgentState,
    SafetyLevel,
    AgentCapability,
)
from greenlang.agents.process_heat.shared.calculation_library import (
    ThermalIQCalculationLibrary,
    CalculationResult,
    UncertaintyBounds,
)
from greenlang.agents.process_heat.shared.coordination import (
    MultiAgentCoordinator,
    CoordinationMessage,
    CoordinationProtocol,
)
from greenlang.agents.process_heat.shared.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    DataLineage,
)
from greenlang.agents.process_heat.shared.audit import (
    AuditLogger,
    AuditEvent,
    AuditLevel,
    ComplianceAuditTrail,
)

__all__ = [
    # Base Agent
    "BaseProcessHeatAgent",
    "AgentState",
    "SafetyLevel",
    "AgentCapability",
    # Calculation Library
    "ThermalIQCalculationLibrary",
    "CalculationResult",
    "UncertaintyBounds",
    # Coordination
    "MultiAgentCoordinator",
    "CoordinationMessage",
    "CoordinationProtocol",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "DataLineage",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditLevel",
    "ComplianceAuditTrail",
]

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"
