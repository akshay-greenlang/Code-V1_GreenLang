"""
SIL Assessment Framework - IEC 61511 Compliant Safety Integrity Level Assessment

This module provides comprehensive tools for Safety Integrity Level (SIL) assessment
per IEC 61511 (Functional safety - Safety instrumented systems for the process industry).

Components:
- LOPAAnalyzer: Layer of Protection Analysis for SIL target determination
- PFDCalculator: Probability of Failure on Demand calculations
- SILClassifier: SIL level classification based on PFD values
- ProofTestScheduler: Proof test interval management
- HardwareFaultTolerance: HFT requirements per IEC 61511

Example:
    >>> from greenlang.safety.sil import LOPAAnalyzer, PFDCalculator, SILClassifier
    >>> lopa = LOPAAnalyzer()
    >>> target_pfd = lopa.calculate_required_pfd(scenario)
    >>> sil_level = SILClassifier.classify_from_pfd(target_pfd)
"""

from greenlang.safety.sil.lopa_analyzer import LOPAAnalyzer, LOPAScenario, LOPAResult
from greenlang.safety.sil.pfd_calculator import (
    PFDCalculator,
    PFDInput,
    PFDResult,
    VotingArchitecture,
)
from greenlang.safety.sil.sil_classifier import (
    SILClassifier,
    SILLevel,
    SILClassificationResult,
)
from greenlang.safety.sil.proof_test_scheduler import (
    ProofTestScheduler,
    ProofTestSchedule,
    ProofTestRecord,
)
from greenlang.safety.sil.hardware_fault_tolerance import (
    HardwareFaultTolerance,
    HFTRequirement,
    HFTAssessment,
)

__all__ = [
    # LOPA
    "LOPAAnalyzer",
    "LOPAScenario",
    "LOPAResult",
    # PFD
    "PFDCalculator",
    "PFDInput",
    "PFDResult",
    "VotingArchitecture",
    # SIL Classification
    "SILClassifier",
    "SILLevel",
    "SILClassificationResult",
    # Proof Test
    "ProofTestScheduler",
    "ProofTestSchedule",
    "ProofTestRecord",
    # HFT
    "HardwareFaultTolerance",
    "HFTRequirement",
    "HFTAssessment",
]

__version__ = "1.0.0"
