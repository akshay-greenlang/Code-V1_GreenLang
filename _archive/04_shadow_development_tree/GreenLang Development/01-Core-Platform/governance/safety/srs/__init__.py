"""
Safety Requirements Specification (SRS) Framework - IEC 61511 Compliant

This module provides tools for creating and managing Safety Requirements
Specifications (SRS) per IEC 61511-1 Clause 10.

Components:
- SRSGenerator: Safety Requirements Specification document generator
- SafeStateManager: Safe state definitions and transitions
- SafetyFunction: Safety function modeling and specification
- ProcessSafetyTime: Process Safety Time (PST) calculator
- DiagnosticCoverage: Diagnostic coverage calculator

Reference: IEC 61511-1 Clause 10 - Safety requirements specification for SIS

Example:
    >>> from greenlang.safety.srs import SRSGenerator, SafetyFunction
    >>> generator = SRSGenerator()
    >>> srs = generator.create_srs(safety_function)
"""

from greenlang.safety.srs.srs_generator import (
    SRSGenerator,
    SRSDocument,
    SRSSection,
)
from greenlang.safety.srs.safe_state_manager import (
    SafeStateManager,
    SafeState,
    SafeStateTransition,
)
from greenlang.safety.srs.safety_function import (
    SafetyFunction,
    SafetyFunctionSpec,
    InputSensor,
    OutputActuator,
)
from greenlang.safety.srs.process_safety_time import (
    ProcessSafetyTimeCalculator,
    PSTInput,
    PSTResult,
)
from greenlang.safety.srs.diagnostic_coverage import (
    DiagnosticCoverageCalculator,
    DCInput,
    DCResult,
)

__all__ = [
    # SRS Generator
    "SRSGenerator",
    "SRSDocument",
    "SRSSection",
    # Safe State
    "SafeStateManager",
    "SafeState",
    "SafeStateTransition",
    # Safety Function
    "SafetyFunction",
    "SafetyFunctionSpec",
    "InputSensor",
    "OutputActuator",
    # Process Safety Time
    "ProcessSafetyTimeCalculator",
    "PSTInput",
    "PSTResult",
    # Diagnostic Coverage
    "DiagnosticCoverageCalculator",
    "DCInput",
    "DCResult",
]

__version__ = "1.0.0"
