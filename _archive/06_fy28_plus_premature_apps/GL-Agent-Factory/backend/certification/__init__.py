"""
GreenLang Agent Certification Framework

This module provides a comprehensive 12-dimension certification framework
for validating agents meet production quality standards.

Dimensions:
    1. Determinism - Byte-identical outputs across runs
    2. Provenance - SHA-256 audit trail verification
    3. Zero-Hallucination - No LLM in calculation path
    4. Accuracy - Golden test pass rate
    5. Source Verification - Traceable emission factors
    6. Unit Consistency - Input/output unit validation
    7. Regulatory Compliance - Standard alignment
    8. Security - No secrets, input sanitization
    9. Performance - Response time and memory
    10. Documentation - Docstrings and API docs
    11. Test Coverage - Code coverage metrics
    12. Production Readiness - Logging and health checks

Example:
    >>> from backend.certification import CertificationEngine
    >>> engine = CertificationEngine()
    >>> result = engine.evaluate_agent(Path("path/to/agent"))
    >>> if result.certified:
    ...     print(f"Agent certified at {result.level} level")
"""

from .engine import CertificationEngine, CertificationResult
from .report import ReportGenerator, CertificationReport

__all__ = [
    "CertificationEngine",
    "CertificationResult",
    "ReportGenerator",
    "CertificationReport",
]

__version__ = "1.0.0"
