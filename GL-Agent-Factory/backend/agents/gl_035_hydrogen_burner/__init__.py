"""
GL-035: Hydrogen Combustion Controller Agent (H2-BURNER)

This package provides the HydrogenBurnerAgent for safe and efficient
hydrogen combustion control in industrial burners.

Key Features:
- Hydrogen combustion safety monitoring
- Flame speed and flashback prevention
- NOx emissions control
- H2 blending ratio optimization
- Complete SHA-256 provenance tracking

Standards Compliance:
- NFPA 86: Standard for Ovens and Furnaces
- ISO 23828: Hydrogen Fuel Systems
- IEC 60079: Explosive Atmospheres

Example Usage:
    >>> from backend.agents.gl_035_hydrogen_burner import (
    ...     HydrogenBurnerAgent,
    ...     HydrogenBurnerInput,
    ... )
    >>> agent = HydrogenBurnerAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Safety Score: {result.safety_score}")
"""

from .agent import (
    HydrogenBurnerAgent,
    HydrogenBurnerInput,
    HydrogenBurnerOutput,
    CombustionData,
    SafetyCheck,
    PACK_SPEC,
)

__all__ = [
    "HydrogenBurnerAgent",
    "HydrogenBurnerInput",
    "HydrogenBurnerOutput",
    "CombustionData",
    "SafetyCheck",
    "PACK_SPEC",
]

__version__ = "1.0.0"
__agent_id__ = "GL-035"
__agent_name__ = "H2-BURNER"
