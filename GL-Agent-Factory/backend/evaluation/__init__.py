"""
GreenLang Agent Evaluation Framework

This module provides comprehensive evaluation and certification capabilities
for GreenLang agents including:
- Golden test execution
- Determinism verification
- Provenance validation
- Certification suite

Example:
    >>> from evaluation import CertificationSuite
    >>> suite = CertificationSuite()
    >>> report = suite.certify_agent("path/to/pack.yaml")
    >>> print(report.certification_status)  # PASS or FAIL

"""

from .golden_test_runner import GoldenTestRunner, GoldenTestResult
from .determinism_verifier import DeterminismVerifier, DeterminismResult
from .certification_suite import CertificationSuite, CertificationReport

__version__ = "1.0.0"

__all__ = [
    'GoldenTestRunner',
    'GoldenTestResult',
    'DeterminismVerifier',
    'DeterminismResult',
    'CertificationSuite',
    'CertificationReport',
]
