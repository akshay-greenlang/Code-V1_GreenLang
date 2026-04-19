"""
Risk Analysis Framework - HAZOP and FMEA Tools

This module provides risk analysis tools for Safety Instrumented Systems:
- HAZOPAnalyzer: Hazard and Operability Study framework
- FMEAAnalyzer: Failure Mode and Effects Analysis tool
- RiskMatrix: Risk severity/likelihood matrix
- SafeguardVerifier: Safeguard verification
- RiskRegister: Risk register management

Reference: IEC 61882 (HAZOP), IEC 60812 (FMEA), ISO 31000

Example:
    >>> from greenlang.safety.risk import HAZOPAnalyzer, RiskMatrix
    >>> hazop = HAZOPAnalyzer()
    >>> risk_matrix = RiskMatrix()
"""

from greenlang.safety.risk.hazop_analyzer import (
    HAZOPAnalyzer,
    HAZOPStudy,
    HAZOPDeviation,
)
from greenlang.safety.risk.fmea_analyzer import (
    FMEAAnalyzer,
    FMEAStudy,
    FailureMode,
)
from greenlang.safety.risk.risk_matrix import (
    RiskMatrix,
    RiskLevel,
    RiskAssessment,
)
from greenlang.safety.risk.safeguard_verifier import (
    SafeguardVerifier,
    Safeguard,
    VerificationResult,
)
from greenlang.safety.risk.risk_register import (
    RiskRegister,
    Risk,
    RiskTreatment,
)

__all__ = [
    # HAZOP
    "HAZOPAnalyzer",
    "HAZOPStudy",
    "HAZOPDeviation",
    # FMEA
    "FMEAAnalyzer",
    "FMEAStudy",
    "FailureMode",
    # Risk Matrix
    "RiskMatrix",
    "RiskLevel",
    "RiskAssessment",
    # Safeguard
    "SafeguardVerifier",
    "Safeguard",
    "VerificationResult",
    # Risk Register
    "RiskRegister",
    "Risk",
    "RiskTreatment",
]

__version__ = "1.0.0"
