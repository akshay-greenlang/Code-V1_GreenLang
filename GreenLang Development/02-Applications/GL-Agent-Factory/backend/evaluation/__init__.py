"""
GreenLang Agent Evaluation Framework

This module provides comprehensive evaluation and certification capabilities
for GreenLang agents including:
- Golden test execution
- Determinism verification
- Provenance validation
- Certification suite
- 12-Dimension Certification Pipeline
- Shadow Mode Testing

Example:
    >>> from evaluation import CertificationSuite
    >>> suite = CertificationSuite()
    >>> report = suite.certify_agent("path/to/pack.yaml")
    >>> print(report.certification_status)  # PASS or FAIL

    # New 12-Dimension Pipeline
    >>> from evaluation.certification import CertificationPipeline
    >>> pipeline = CertificationPipeline()
    >>> report = pipeline.certify_agent(agent, "path/to/pack.yaml")
    >>> print(f"Level: {report.certification_level}")

    # Shadow Mode Testing
    >>> from evaluation.shadow_mode import ShadowModeRunner, ShadowConfig
    >>> config = ShadowConfig(mode=ShadowMode.PARALLEL, comparison_tolerance=0.001)
    >>> runner = ShadowModeRunner(config)
    >>> report = await runner.run_parallel_shadow(baseline, candidate, traffic)
    >>> print(f"Match rate: {report.match_rate:.2%}")

"""

from .golden_test_runner import GoldenTestRunner, GoldenTestResult
from .determinism_verifier import DeterminismVerifier, DeterminismResult
from .certification_suite import CertificationSuite, CertificationReport

# Import certification pipeline components
from .certification import (
    CertificationPipeline,
    CertificationConfig,
    CertificationLevel,
    DimensionResult,
    TestGenerator,
    ReportGenerator,
    CertificationRegistry,
    CertifiedAgent,
    CertificationStatus,
)

# Import shadow mode components
from .shadow_mode import (
    ShadowMode,
    ShadowConfig,
    TrafficRecord,
    ShadowResult,
    ShadowReport,
    ShadowModeRunner,
    TrafficRecorder,
    OutputComparator,
    ShadowModeEvaluator,
    create_shadow_config,
    run_shadow_test,
)

__version__ = "1.0.0"

__all__ = [
    # Original exports
    'GoldenTestRunner',
    'GoldenTestResult',
    'DeterminismVerifier',
    'DeterminismResult',
    'CertificationSuite',
    'CertificationReport',
    # Certification pipeline exports
    'CertificationPipeline',
    'CertificationConfig',
    'CertificationLevel',
    'DimensionResult',
    'TestGenerator',
    'ReportGenerator',
    'CertificationRegistry',
    'CertifiedAgent',
    'CertificationStatus',
    # Shadow mode exports
    'ShadowMode',
    'ShadowConfig',
    'TrafficRecord',
    'ShadowResult',
    'ShadowReport',
    'ShadowModeRunner',
    'TrafficRecorder',
    'OutputComparator',
    'ShadowModeEvaluator',
    'create_shadow_config',
    'run_shadow_test',
]
