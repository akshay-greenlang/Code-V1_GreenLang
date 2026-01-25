"""
GreenLang Data Engineering Module
=================================

Enterprise-grade data integration infrastructure for 100K+ emission factors.

Modules:
- connectors: ERP system connectors (SAP, Oracle, Workday)
- etl: Extract-Transform-Load pipeline components
- parsers: Multi-format file parsers (CSV, Excel, JSON, XML)
- schemas: Database schemas and data contracts
- validation: Data validation rules engine
- quality: Data quality scoring framework
- pipelines: Orchestrated data pipelines

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from greenlang.data_engineering.schemas.emission_factor_schema import (
    EmissionFactorSchema,
    EmissionFactorVersion,
    EmissionFactorQuality,
    EmissionFactorSource,
)
from greenlang.data_engineering.validation.rules_engine import (
    ValidationRulesEngine,
    ValidationRule,
    ValidationResult,
)
from greenlang.data_engineering.quality.scoring import (
    DataQualityScorer,
    QualityDimension,
    QualityScore,
)

__version__ = "1.0.0"
__all__ = [
    "EmissionFactorSchema",
    "EmissionFactorVersion",
    "EmissionFactorQuality",
    "EmissionFactorSource",
    "ValidationRulesEngine",
    "ValidationRule",
    "ValidationResult",
    "DataQualityScorer",
    "QualityDimension",
    "QualityScore",
]
