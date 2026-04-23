"""
GreenLang Agent: GL-012_SteamQual

Steam Quality Controller - Real-time steam quality monitoring, dryness fraction
estimation, and carryover risk assessment for industrial steam systems.

This module provides:
- Real-time estimation of steam dryness fraction (x) and moisture carryover risk
- Closed-loop and supervisory control recommendations for quality management
- Quality event detection, root-cause analytics, and operator guidance
- Measurement governance: sensor selection, calibration, and uncertainty reporting

Consolidated into GL-003 UNIFIEDSTEAM as a domain module.

Version: 1.0.0
Category: steam
Type: hybrid (calculator + optimizer + analyzer)

Standards Compliance:
- ASME PTC 19.11 Steam Quality
- IAPWS-IF97 Steam Tables
- GreenLang Framework v1.0

Author: GreenLang AI Agent Workforce
"""

__version__ = "1.0.0"
__agent_id__ = "GL-012"
__agent_name__ = "SteamQual"
__full_name__ = "Steam Quality Controller"

from .core import SteamQualConfig, SteamQualOrchestrator
from .models import (
    SteamMeasurement,
    QualityEstimate,
    CarryoverRiskAssessment,
    QualityState,
    QualityConstraints,
    QualityEvent,
    SteamState,
    EventType,
    Severity,
)

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "__full_name__",
    # Core
    "SteamQualConfig",
    "SteamQualOrchestrator",
    # Models
    "SteamMeasurement",
    "QualityEstimate",
    "CarryoverRiskAssessment",
    "QualityState",
    "QualityConstraints",
    "QualityEvent",
    "SteamState",
    "EventType",
    "Severity",
]
