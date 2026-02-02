"""
GL-031: Furnace Guardian Agent (FURNACE-GUARDIAN)

This package provides the FurnaceGuardianAgent for furnace safety monitoring
in industrial process heat systems.

Key Features:
- Interlock validation per NFPA 86, API 560, EN 746
- Purge verification and timing
- Flame supervision with UV/IR scanner validation
- Temperature and pressure limit monitoring
- Complete SHA-256 provenance tracking

Standards Compliance:
- NFPA 86: Standard for Ovens and Furnaces
- API 560: Fired Heaters for General Refinery Service
- EN 746: Industrial Thermoprocessing Equipment

Example Usage:
    >>> from backend.agents.gl_031_furnace_guardian import (
    ...     FurnaceGuardianAgent,
    ...     FurnaceGuardianInput,
    ... )
    >>> agent = FurnaceGuardianAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Safety Score: {result.safety_score}")
"""

from .agent import (
    FurnaceGuardianAgent,
    FurnaceGuardianInput,
    FurnaceGuardianOutput,
    InterlockStatus,
    FlameStatus,
    TemperatureReading,
    PressureReading,
    SafetyViolation,
    CorrectiveAction,
    ComplianceStatus,
    PurgeData,
    PACK_SPEC,
)

from .models import (
    RiskLevel,
    InterlockType,
    ViolationSeverity,
    FlameDetectorType,
    FurnaceType,
    ComplianceStandard,
    PurgeStatus,
)

__all__ = [
    "FurnaceGuardianAgent",
    "FurnaceGuardianInput",
    "FurnaceGuardianOutput",
    "InterlockStatus",
    "FlameStatus",
    "TemperatureReading",
    "PressureReading",
    "SafetyViolation",
    "CorrectiveAction",
    "ComplianceStatus",
    "PurgeData",
    "RiskLevel",
    "InterlockType",
    "ViolationSeverity",
    "FlameDetectorType",
    "FurnaceType",
    "ComplianceStandard",
    "PurgeStatus",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-031"
__agent_name__ = "FURNACE-GUARDIAN"
