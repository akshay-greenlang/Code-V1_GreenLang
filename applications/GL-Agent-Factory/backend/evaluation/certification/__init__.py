"""
GreenLang Agent Certification Pipeline

A comprehensive 12-dimension certification system for GreenLang AI agents.

Dimensions:
1. Technical Accuracy - Formula correctness, unit handling
2. Data Credibility - Source validation, provenance tracking
3. Safety Compliance - NFPA/IEC/OSHA compliance checks
4. Regulatory Alignment - Standards adherence verification
5. Uncertainty Quantification - Error bounds, confidence intervals
6. Explainability - Decision transparency, reasoning traces
7. Performance - Latency, throughput benchmarks
8. Robustness - Edge cases, error handling
9. Security - Input validation, injection prevention
10. Auditability - Logging, reproducibility
11. Maintainability - Code quality, documentation
12. Operability - Monitoring, alerting readiness

Example:
    >>> from evaluation.certification import CertificationPipeline
    >>> pipeline = CertificationPipeline()
    >>> report = pipeline.certify_agent(agent, pack_yaml_path)
    >>> print(f"Certification: {report.certification_level}")
    >>> print(f"Overall Score: {report.overall_score}/100")

"""

from .certification_pipeline import (
    CertificationPipeline,
    CertificationConfig,
    DimensionResult,
    CertificationReport,
    CertificationLevel,
)
from .test_generators import (
    TestGenerator,
    GoldenTestGenerator,
    BoundaryTestGenerator,
    FuzzTestGenerator,
    AdversarialTestGenerator,
)
from .reports import (
    ReportGenerator,
    CertificationScorecard,
    ComplianceMatrix,
    RecommendationEngine,
)
from .registry import (
    CertificationRegistry,
    CertifiedAgent,
    CertificationStatus,
)

# Dimension evaluators
from .dimensions import (
    TechnicalAccuracyEvaluator,
    DataCredibilityEvaluator,
    SafetyComplianceEvaluator,
    RegulatoryAlignmentEvaluator,
    UncertaintyQuantificationEvaluator,
    ExplainabilityEvaluator,
    PerformanceEvaluator,
    RobustnessEvaluator,
    SecurityEvaluator,
    AuditabilityEvaluator,
    MaintainabilityEvaluator,
    OperabilityEvaluator,
)

__version__ = "1.0.0"

__all__ = [
    # Main pipeline
    "CertificationPipeline",
    "CertificationConfig",
    "DimensionResult",
    "CertificationReport",
    "CertificationLevel",
    # Test generators
    "TestGenerator",
    "GoldenTestGenerator",
    "BoundaryTestGenerator",
    "FuzzTestGenerator",
    "AdversarialTestGenerator",
    # Reporting
    "ReportGenerator",
    "CertificationScorecard",
    "ComplianceMatrix",
    "RecommendationEngine",
    # Registry
    "CertificationRegistry",
    "CertifiedAgent",
    "CertificationStatus",
    # Dimension evaluators
    "TechnicalAccuracyEvaluator",
    "DataCredibilityEvaluator",
    "SafetyComplianceEvaluator",
    "RegulatoryAlignmentEvaluator",
    "UncertaintyQuantificationEvaluator",
    "ExplainabilityEvaluator",
    "PerformanceEvaluator",
    "RobustnessEvaluator",
    "SecurityEvaluator",
    "AuditabilityEvaluator",
    "MaintainabilityEvaluator",
    "OperabilityEvaluator",
]
