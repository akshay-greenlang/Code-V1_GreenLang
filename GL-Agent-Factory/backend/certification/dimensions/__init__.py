"""
Certification Dimensions Package

This package contains the 12 certification dimension evaluators
that comprise the GreenLang agent certification framework.

Each dimension evaluator follows a standard interface:
    - evaluate(agent_path: Path, agent: Any, config: Dict) -> DimensionResult
    - get_remediation_suggestions(result: DimensionResult) -> List[str]
"""

from .base import BaseDimension, DimensionResult, DimensionStatus
from .d01_determinism import DeterminismDimension
from .d02_provenance import ProvenanceDimension
from .d03_zero_hallucination import ZeroHallucinationDimension
from .d04_accuracy import AccuracyDimension
from .d05_source_verification import SourceVerificationDimension
from .d06_unit_consistency import UnitConsistencyDimension
from .d07_regulatory import RegulatoryComplianceDimension
from .d08_security import SecurityDimension
from .d09_performance import PerformanceDimension
from .d10_documentation import DocumentationDimension
from .d11_coverage import CoverageDimension
from .d12_production import ProductionReadinessDimension

ALL_DIMENSIONS = [
    DeterminismDimension,
    ProvenanceDimension,
    ZeroHallucinationDimension,
    AccuracyDimension,
    SourceVerificationDimension,
    UnitConsistencyDimension,
    RegulatoryComplianceDimension,
    SecurityDimension,
    PerformanceDimension,
    DocumentationDimension,
    CoverageDimension,
    ProductionReadinessDimension,
]

__all__ = [
    "BaseDimension",
    "DimensionResult",
    "DimensionStatus",
    "DeterminismDimension",
    "ProvenanceDimension",
    "ZeroHallucinationDimension",
    "AccuracyDimension",
    "SourceVerificationDimension",
    "UnitConsistencyDimension",
    "RegulatoryComplianceDimension",
    "SecurityDimension",
    "PerformanceDimension",
    "DocumentationDimension",
    "CoverageDimension",
    "ProductionReadinessDimension",
    "ALL_DIMENSIONS",
]
