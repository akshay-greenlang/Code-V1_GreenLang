"""
GL-005 CombustionControlAgent

Real-time automated combustion control agent for consistent heat output and stability.

Agent ID: GL-005
Version: 1.0.0
Category: Combustion Control
Type: Real-time Controller

Key Features:
- Real-time PID control loops (<100ms cycle time)
- Fuel and air flow control
- O2 trim optimization
- Heat output stability control
- Multi-variable control with feedforward
- Safety interlocks and fail-safes
- DCS/PLC integration
- Zero-hallucination deterministic control

Primary Function:
Automated control of combustion processes for consistent heat output

Inputs:
- Fuel flow rate
- Air flow rate
- Temperature measurements (flame, furnace, flue gas)
- Pressure measurements (fuel, air, furnace)
- O2, CO, CO2, NOx from analyzers

Outputs:
- Real-time combustion adjustments (fuel/air setpoints)
- Stability metrics
- Control performance data
- Safety status

Integrations:
- DCS (Distributed Control System)
- PLC (Programmable Logic Controller)
- Combustion analyzers
- SCADA/OPC UA
- MQTT for real-time telemetry
"""

__version__ = "1.0.0"
__agent_id__ = "GL-005"
__agent_name__ = "CombustionControlAgent"

from .combustion_control_orchestrator import (
    CombustionControlOrchestrator,
    CombustionState,
    ControlAction,
    StabilityMetrics,
    SafetyInterlocks
)

from .config import settings

from .tools import (
    TOOL_REGISTRY,
    get_tool,
    list_tools,
    get_tool_schema,
    validate_tool_input,
    validate_tool_output,
    get_all_schemas
)

__all__ = [
    # Version info
    '__version__',
    '__agent_id__',
    '__agent_name__',

    # Main orchestrator
    'CombustionControlOrchestrator',

    # Data models
    'CombustionState',
    'ControlAction',
    'StabilityMetrics',
    'SafetyInterlocks',

    # Configuration
    'settings',

    # Tool registry
    'TOOL_REGISTRY',
    'get_tool',
    'list_tools',
    'get_tool_schema',
    'validate_tool_input',
    'validate_tool_output',
    'get_all_schemas'
]
