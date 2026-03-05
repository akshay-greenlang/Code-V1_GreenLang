"""
GL-CDP-APP v1.0 -- CDP Climate Change Disclosure Platform Services

This package provides configuration, domain models, and 13 service engines for
implementing the CDP Climate Change Questionnaire disclosure management platform
with auto-population from 30 MRV agents, scoring simulation (D- through A),
gap analysis, sector benchmarking, and 1.5C transition plan building.

Engines (13 total):
    config: Enumerations, scoring weights, thresholds, and application settings.
    models: Pydantic domain models for all entities.
    questionnaire_engine: Full CDP questionnaire (13 modules, 200+ questions).
    response_manager: Response lifecycle with versioning and review workflow.
    scoring_simulator: 17-category CDP scoring engine with what-if analysis.
    data_connector: MRV agent integration for Scope 1/2/3 auto-population.
    gap_analysis_engine: Gap identification with recommendations and priority.
    benchmarking_engine: Sector and regional peer comparison.
    supply_chain_module: Supplier engagement and emissions aggregation.
    transition_plan_engine: 1.5C transition plan builder with SBTi alignment.
    verification_tracker: Third-party verification status management.
    historical_tracker: Year-over-year score progression and trend analysis.
    report_generator: PDF, Excel, XML/ORS report generation.
    setup: Platform facade composing all engines.

Standard: CDP Climate Change Questionnaire (2025/2026 Integrated Format)
Aligned: IFRS S2, ESRS E1, TCFD, GRI 305, SBTi, GHG Protocol, ISO 14064-1
MRV Base: 30 production agents (750K+ lines)
"""

__version__ = "1.0.0"
__standard__ = "CDP Climate Change Questionnaire 2025/2026"

from .config import CDPAppConfig
from .questionnaire_engine import QuestionnaireEngine
from .response_manager import ResponseManager
from .scoring_simulator import ScoringSimulator
from .data_connector import DataConnector
from .gap_analysis_engine import GapAnalysisEngine
from .benchmarking_engine import BenchmarkingEngine
from .supply_chain_module import SupplyChainModule
from .transition_plan_engine import TransitionPlanEngine
from .verification_tracker import VerificationTracker
from .historical_tracker import HistoricalTracker
from .report_generator import ReportGenerator
from .setup import CDPPlatform

__all__ = [
    "__version__",
    "__standard__",
    "CDPAppConfig",
    "QuestionnaireEngine",
    "ResponseManager",
    "ScoringSimulator",
    "DataConnector",
    "GapAnalysisEngine",
    "BenchmarkingEngine",
    "SupplyChainModule",
    "TransitionPlanEngine",
    "VerificationTracker",
    "HistoricalTracker",
    "ReportGenerator",
    "CDPPlatform",
]
