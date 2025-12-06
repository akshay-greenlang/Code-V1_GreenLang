"""
Emergency Shutdown (ESD) Framework - IEC 61511 Compliant ESD System

This module provides Emergency Shutdown system components and interfaces
per IEC 61511 for Safety Instrumented Systems.

Components:
- ESDInterface: ESD system interface specification
- PriorityManager: ESD > DCS > Agent priority management
- ResponseValidator: <1s response time validation
- BypassManager: Bypass management per IEC 61511
- ESDSimulator: ESD simulation mode for testing

Reference: IEC 61511-1 Clause 11, IEC 61508-2

Example:
    >>> from greenlang.safety.esd import ESDInterface, PriorityManager
    >>> esd = ESDInterface()
    >>> priority = PriorityManager()
"""

from greenlang.safety.esd.esd_interface import (
    ESDInterface,
    ESDCommand,
    ESDStatus,
)
from greenlang.safety.esd.priority_manager import (
    PriorityManager,
    CommandPriority,
    PriorityResult,
)
from greenlang.safety.esd.response_validator import (
    ResponseValidator,
    ResponseTest,
    ResponseResult,
)
from greenlang.safety.esd.bypass_manager import (
    BypassManager,
    BypassRequest,
    BypassRecord,
)
from greenlang.safety.esd.esd_simulator import (
    ESDSimulator,
    SimulationConfig,
    SimulationResult,
)

__all__ = [
    # ESD Interface
    "ESDInterface",
    "ESDCommand",
    "ESDStatus",
    # Priority Manager
    "PriorityManager",
    "CommandPriority",
    "PriorityResult",
    # Response Validator
    "ResponseValidator",
    "ResponseTest",
    "ResponseResult",
    # Bypass Manager
    "BypassManager",
    "BypassRequest",
    "BypassRecord",
    # ESD Simulator
    "ESDSimulator",
    "SimulationConfig",
    "SimulationResult",
]

__version__ = "1.0.0"
