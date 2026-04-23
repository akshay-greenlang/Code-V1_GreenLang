"""
GreenLang Agent Framework
=========================

Enterprise framework for building, scoring, and deploying
GreenLang AI agents for industrial process optimization.

This framework provides:
- Standardized agent templates and base classes
- Quality scoring and assessment tools
- Shared utilities for deterministic calculations
- Agent scaffolding generator
- Compliance validation against global standards

Standards Compliance:
- ISO/IEC 42001:2023 (AI Management System)
- ISO/IEC 23894:2023 (AI Risk Management)
- NIST AI RMF 1.0 (Risk Management Framework)
- EU AI Act requirements for high-risk systems
- GreenLang Specification v1.0

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Technologies"

from .standards import (
    QualityStandard,
    QualityDimension,
    ComplianceLevel,
    AgentClass,
    DomainCategory,
)
from .scoring import (
    AgentScorer,
    ScoreReport,
)
from .shared import (
    ProvenanceTracker,
    DeterministicCalculator,
    UnitConverter,
    ValidationEngine,
)

__all__ = [
    # Standards
    "QualityStandard",
    "ComplianceLevel",
    "AgentClass",
    "DomainCategory",
    # Scoring
    "AgentScorer",
    "ScoreReport",
    "QualityDimension",
    # Shared
    "ProvenanceTracker",
    "DeterministicCalculator",
    "UnitConverter",
    "ValidationEngine",
]
