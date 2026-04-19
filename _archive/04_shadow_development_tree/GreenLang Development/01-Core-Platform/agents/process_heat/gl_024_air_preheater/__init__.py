"""
GL-024 AIRPREHEATER - Air Preheater Optimizer Agent

Optimizes air preheater performance for maximum heat recovery from flue gas
while maintaining cold-end protection and minimizing leakage losses.

Key Capabilities:
    - Heat transfer analysis (epsilon-NTU, LMTD methods)
    - Leakage detection and quantification (O2 differential method)
    - Cold-end corrosion protection (acid dew point monitoring)
    - Fouling management and soot blowing optimization
    - Performance optimization with SHAP/LIME explainability

Engineering Standards:
    - ASME PTC 4.3 - Air Heaters Performance Test Code
    - API 560 - Fired Heaters for General Refinery Service
    - NFPA 85/86 - Boiler and Combustion Systems Hazards Code
    - ISO 12759 - Fans for Industrial Applications

Air Preheater Types Supported:
    - Regenerative (Ljungström rotating, Rothemuhle stationary)
    - Recuperative (tubular, plate)
    - Heat pipe

Market Size: $4B
Priority: P2 (Medium)
Timeline: Q3 2026
Status: Implemented

Example:
    >>> from greenlang.agents.process_heat.gl_024_air_preheater import (
    ...     AirPreheaterAgent,
    ...     AirPreheaterConfig,
    ...     AirPreheaterInput,
    ... )
    >>>
    >>> agent = AirPreheaterAgent(AirPreheaterConfig())
    >>> result = agent.optimize(input_data)
"""

from .agent import AirPreheaterAgent, create_agent
from .config import AirPreheaterConfig, AirPreheaterThresholds
from .schemas import (
    AirPreheaterInput,
    AirPreheaterOutput,
    AirPreheaterType,
    HeatTransferAnalysis,
    LeakageAnalysis,
    ColdEndProtection,
    FoulingAnalysis,
    OptimizationResult,
)
from .calculations import AirPreheaterCalculator
from .explainability import (
    LIMEAirPreheaterExplainer,
    create_explainer,
    ExplainerConfig,
)
from .provenance import ProvenanceTracker

__all__ = [
    # Agent
    "AirPreheaterAgent",
    "create_agent",
    # Configuration
    "AirPreheaterConfig",
    "AirPreheaterThresholds",
    # Schemas
    "AirPreheaterInput",
    "AirPreheaterOutput",
    "AirPreheaterType",
    "HeatTransferAnalysis",
    "LeakageAnalysis",
    "ColdEndProtection",
    "FoulingAnalysis",
    "OptimizationResult",
    # Calculations
    "AirPreheaterCalculator",
    # Explainability
    "LIMEAirPreheaterExplainer",
    "create_explainer",
    "ExplainerConfig",
    # Provenance
    "ProvenanceTracker",
]

__version__ = "1.0.0"
__agent_id__ = "GL-024"
__agent_name__ = "AIRPREHEATER"
