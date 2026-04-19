"""
Certification Dimension Evaluators

This module provides 12 dimension evaluators for comprehensive agent certification.
Each evaluator scores agents on a specific quality dimension (0-100 scale).

Dimensions:
    - Technical Accuracy: Formula correctness, unit handling
    - Data Credibility: Source validation, provenance
    - Safety Compliance: NFPA/IEC/OSHA compliance
    - Regulatory Alignment: Standards adherence
    - Uncertainty Quantification: Error bounds, confidence
    - Explainability: Decision transparency
    - Performance: Latency, throughput benchmarks
    - Robustness: Edge cases, error handling
    - Security: Input validation, injection prevention
    - Auditability: Logging, reproducibility
    - Maintainability: Code quality, documentation
    - Operability: Monitoring, alerting readiness

"""

from .technical_accuracy import TechnicalAccuracyEvaluator
from .data_credibility import DataCredibilityEvaluator
from .safety_compliance import SafetyComplianceEvaluator
from .regulatory_alignment import RegulatoryAlignmentEvaluator
from .uncertainty_quantification import UncertaintyQuantificationEvaluator
from .explainability import ExplainabilityEvaluator
from .performance import PerformanceEvaluator
from .robustness import RobustnessEvaluator
from .security import SecurityEvaluator
from .auditability import AuditabilityEvaluator
from .maintainability import MaintainabilityEvaluator
from .operability import OperabilityEvaluator

__all__ = [
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

# Dimension weights (must sum to 1.0)
DIMENSION_WEIGHTS = {
    "technical_accuracy": 0.15,
    "data_credibility": 0.12,
    "safety_compliance": 0.12,
    "regulatory_alignment": 0.10,
    "uncertainty_quantification": 0.08,
    "explainability": 0.08,
    "performance": 0.08,
    "robustness": 0.08,
    "security": 0.07,
    "auditability": 0.05,
    "maintainability": 0.04,
    "operability": 0.03,
}

# Minimum thresholds per dimension (must score above these to pass)
DIMENSION_THRESHOLDS = {
    "technical_accuracy": 85.0,
    "data_credibility": 80.0,
    "safety_compliance": 90.0,  # Higher threshold for safety
    "regulatory_alignment": 85.0,
    "uncertainty_quantification": 70.0,
    "explainability": 70.0,
    "performance": 75.0,
    "robustness": 75.0,
    "security": 80.0,
    "auditability": 70.0,
    "maintainability": 60.0,
    "operability": 60.0,
}
