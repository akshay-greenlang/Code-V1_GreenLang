"""
GreenLang Safety Boundary Infrastructure

This package provides IEC 61511 and NFPA compliant safety infrastructure
for process heat agents. All safety calculations are:

- Deterministic: Same input always produces same output
- Reproducible: Complete provenance tracking with SHA-256 hashes
- Auditable: Full audit trail for regulatory compliance
- Zero-Hallucination: NO LLM in safety calculation path

Standards Compliance:
    - IEC 61508: Functional Safety (SIL levels)
    - IEC 61511: Safety Instrumented Systems for Process Industries
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces

Modules:
    sil_calculator: Safety Integrity Level determination per IEC 61508
    lopa_analyzer: Layer of Protection Analysis per IEC 61511
    sis_interface: Safety Instrumented System interface
    trip_logic: Safety trip logic validation and verification
    nfpa_compliance: NFPA 85/86 compliance verification
    boundaries: Safety boundary enforcement and limit checking

SIL Levels (per IEC 61508):
    - SIL 1: PFD 10^-1 to 10^-2 (0.1 to 0.01)
    - SIL 2: PFD 10^-2 to 10^-3 (0.01 to 0.001)
    - SIL 3: PFD 10^-3 to 10^-4 (0.001 to 0.0001)
    - SIL 4: PFD 10^-4 to 10^-5 (0.0001 to 0.00001)

Example:
    >>> from engines.safety import SILCalculator, LOPAAnalyzer, SafetyBoundary
    >>>
    >>> # Calculate SIL level for a safety function
    >>> sil_calc = SILCalculator()
    >>> sil_level = sil_calc.calculate_sil_level(
    ...     consequence_severity=3,
    ...     likelihood_without_protection=0.1,
    ...     target_risk_level=1e-4
    ... )
    >>> print(f"Required SIL: {sil_level}")
    >>>
    >>> # Perform LOPA analysis
    >>> lopa = LOPAAnalyzer()
    >>> result = lopa.analyze_scenario(
    ...     initiating_event=InitiatingEvent(name="Valve failure", frequency=0.1),
    ...     independent_protection_layers=[...],
    ...     target_mitigated_frequency=1e-5
    ... )
    >>>
    >>> # Enforce safety boundaries
    >>> boundary = SafetyBoundary(limits={
    ...     "temperature": (0.0, 1200.0),
    ...     "pressure": (0.0, 15.0),
    ... })
    >>> status = boundary.check_value("temperature", 850.0)

CRITICAL: All safety calculations in this module are DETERMINISTIC.
NO LLM calls are permitted in the safety calculation path.
"""

from .sil_calculator import (
    SILCalculator,
    SILLevel,
    SafetyComponent,
    SILCalculationResult,
    VotingArchitecture,
    ComponentType,
)
from .lopa_analyzer import (
    LOPAAnalyzer,
    LOPAResult,
    InitiatingEvent,
    IPL,
    IPLType,
    ConditionalModifier,
    LOPAScenario,
)
from .sis_interface import (
    SISInterface,
    SISConfig,
    SISStatus,
    SafetyFunction,
    SIFStatus,
    DiagnosticResult,
    ProofTestResult,
)
from .trip_logic import (
    TripLogicValidator,
    TripCondition,
    TripAction,
    TripLogicResult,
    LogicType,
    TripPriority,
)
from .nfpa_compliance import (
    NFPAComplianceChecker,
    NFPAStandard,
    ComplianceResult,
    ComplianceStatus,
    NFPARequirement,
    FurnaceType,
    BurnerType,
)
from .boundaries import (
    SafetyBoundary,
    SafetyStatus,
    BoundaryViolation,
    SafetyLimit,
    LimitType,
    EnforcementAction,
)

__all__ = [
    # SIL Calculator
    "SILCalculator",
    "SILLevel",
    "SafetyComponent",
    "SILCalculationResult",
    "VotingArchitecture",
    "ComponentType",
    # LOPA Analyzer
    "LOPAAnalyzer",
    "LOPAResult",
    "InitiatingEvent",
    "IPL",
    "IPLType",
    "ConditionalModifier",
    "LOPAScenario",
    # SIS Interface
    "SISInterface",
    "SISConfig",
    "SISStatus",
    "SafetyFunction",
    "SIFStatus",
    "DiagnosticResult",
    "ProofTestResult",
    # Trip Logic
    "TripLogicValidator",
    "TripCondition",
    "TripAction",
    "TripLogicResult",
    "LogicType",
    "TripPriority",
    # NFPA Compliance
    "NFPAComplianceChecker",
    "NFPAStandard",
    "ComplianceResult",
    "ComplianceStatus",
    "NFPARequirement",
    "FurnaceType",
    "BurnerType",
    # Boundaries
    "SafetyBoundary",
    "SafetyStatus",
    "BoundaryViolation",
    "SafetyLimit",
    "LimitType",
    "EnforcementAction",
]

__version__ = "1.0.0"
