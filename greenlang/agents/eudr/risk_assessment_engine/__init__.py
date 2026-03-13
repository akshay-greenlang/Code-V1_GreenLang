# -*- coding: utf-8 -*-
"""
AGENT-EUDR-028: Risk Assessment Engine

Calculates composite risk scores, evaluates EUDR Article 10(2) criteria,
classifies risk levels, and generates risk assessment reports for due
diligence statements. Orchestrates seven processing engines across the
full risk assessment lifecycle:

    Engine 1: Composite Risk Calculator      -- Multi-dimensional weighted scoring
    Engine 2: Risk Factor Aggregator         -- Factor normalization and aggregation
    Engine 3: Country Benchmark Engine       -- EU-published benchmark comparison
    Engine 4: Article 10 Criteria Evaluator  -- 7 Article 10(2) criteria evaluation
    Engine 5: Risk Classification Engine     -- Negligible/Low/Standard/High/Critical
    Engine 6: Risk Trend Analyzer            -- Temporal trend and drift detection
    Engine 7: Risk Report Generator          -- Structured assessment report assembly

Package Structure:
    Core Modules:
        - models.py          -- 10 enums, 15+ core models
        - config.py          -- ~60 environment variables with GL_EUDR_RAE_ prefix
        - provenance.py      -- SHA-256 chain hashing
        - metrics.py         -- 18 Prometheus metrics with gl_eudr_rae_ prefix

    Processing Engines:
        - composite_risk_calculator.py       -- Weighted composite scoring
        - risk_factor_aggregator.py          -- Factor normalization and aggregation
        - country_benchmark_engine.py        -- EU benchmark comparison
        - article10_criteria_evaluator.py    -- Article 10(2) criteria evaluation
        - risk_classification_engine.py      -- Risk level classification
        - risk_trend_analyzer.py             -- Trend analysis and drift detection
        - risk_report_generator.py           -- Assessment report generation

    Service Facade:
        - setup.py           -- RiskAssessmentEngineService facade
        - api.py             -- FastAPI router (12 endpoints)

Commodities Supported:
    cattle, cocoa, coffee, oil_palm, rubber, soya, wood

Regulatory References:
    - EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
    - Article 10(1): Risk assessment obligation
    - Article 10(2): 7 criteria for risk evaluation
    - Article 13: Simplified due diligence for low-risk countries
    - Article 29: Country benchmarking system (low/standard/high)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 Risk Assessment Engine (GL-EUDR-RAE-028)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__: list[str] = [
    "__version__",
    # Service facade
    "RiskAssessmentEngineService",
    # Configuration
    "RiskAssessmentEngineConfig",
    "get_config",
    # Engines
    "CompositeRiskCalculator",
    "RiskFactorAggregator",
    "CountryBenchmarkEngine",
    "Article10CriteriaEvaluator",
    "RiskClassificationEngine",
    "RiskTrendAnalyzer",
    "RiskReportGenerator",
    # Provenance
    "ProvenanceTracker",
    # Models (enums)
    "RiskDimension",
    "RiskLevel",
    "RiskAssessmentStatus",
    "CountryBenchmarkLevel",
    "Article10Criterion",
    "CriterionResult",
    "TrendDirection",
    "OverrideReason",
    "EUDRCommodity",
    # Models (core)
    "RiskFactorInput",
    "CompositeRiskScore",
    "Article10CriteriaResult",
    "RiskAssessmentReport",
    "RiskAssessmentOperation",
    "CountryBenchmark",
]


def _lazy_import(name: str) -> object:
    """Lazy import to avoid circular imports at module load time.

    Args:
        name: Name of the attribute to import.

    Returns:
        The imported object.

    Raises:
        AttributeError: If the name is not in __all__.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Service facade
    if name == "RiskAssessmentEngineService":
        from greenlang.agents.eudr.risk_assessment_engine.setup import (
            RiskAssessmentEngineService,
        )
        return RiskAssessmentEngineService

    # Configuration
    if name == "RiskAssessmentEngineConfig":
        from greenlang.agents.eudr.risk_assessment_engine.config import (
            RiskAssessmentEngineConfig,
        )
        return RiskAssessmentEngineConfig
    if name == "get_config":
        from greenlang.agents.eudr.risk_assessment_engine.config import (
            get_config,
        )
        return get_config

    # Engines
    engine_map = {
        "CompositeRiskCalculator": (
            "composite_risk_calculator", "CompositeRiskCalculator"
        ),
        "RiskFactorAggregator": (
            "risk_factor_aggregator", "RiskFactorAggregator"
        ),
        "CountryBenchmarkEngine": (
            "country_benchmark_engine", "CountryBenchmarkEngine"
        ),
        "Article10CriteriaEvaluator": (
            "article10_criteria_evaluator", "Article10CriteriaEvaluator"
        ),
        "RiskClassificationEngine": (
            "risk_classification_engine", "RiskClassificationEngine"
        ),
        "RiskTrendAnalyzer": (
            "risk_trend_analyzer", "RiskTrendAnalyzer"
        ),
        "RiskReportGenerator": (
            "risk_report_generator", "RiskReportGenerator"
        ),
    }
    if name in engine_map:
        module_name, class_name = engine_map[name]
        import importlib
        mod = importlib.import_module(
            f"greenlang.agents.eudr.risk_assessment_engine.{module_name}"
        )
        return getattr(mod, class_name)

    # Provenance
    if name == "ProvenanceTracker":
        from greenlang.agents.eudr.risk_assessment_engine.provenance import (
            ProvenanceTracker,
        )
        return ProvenanceTracker

    # All models are in models.py
    from greenlang.agents.eudr.risk_assessment_engine import models
    if hasattr(models, name):
        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.risk_assessment_engine import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.
    """
    return _lazy_import(name)
