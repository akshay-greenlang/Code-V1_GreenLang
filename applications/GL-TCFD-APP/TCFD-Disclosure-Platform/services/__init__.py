"""
GL-TCFD-APP v1.0 -- TCFD Disclosure & Scenario Analysis Platform Services

This package provides configuration, domain models, and 5 service engines plus
setup facade for implementing the TCFD (Task Force on Climate-related Financial
Disclosures) four-pillar framework with ISSB/IFRS S2 cross-walk, scenario
analysis (IEA/NGFS), gap analysis, and AI-driven recommendations.

Engines (5 service engines + config + models + setup):
    config: Enumerations, scenario parameters, risk matrices, and app settings.
    models: Pydantic v2 domain models for all TCFD entities.
    disclosure_generator: 11 TCFD recommended disclosures lifecycle management.
    issb_crosswalk_engine: TCFD-to-IFRS S2 mapping, gaps, and migration pathway.
    gap_analysis_engine: 40-dimension maturity assessment and action planning.
    recommendation_engine: Prioritized improvement recommendations.
    data_quality_engine: 4-dimension data quality scoring and validation.
    setup: Platform facade composing all engines with FastAPI app factory.

Standard: TCFD Final Report (June 2017)
ISSB Alignment: IFRS S2 Climate-related Disclosures (June 2023)
Scenarios: IEA (NZE, APS, STEPS) + NGFS (Current Policies, Delayed, Below 2C, Divergent NZ)
Regulatory Jurisdictions: UK FCA, EU CSRD, US SEC, Japan FSA, Singapore SGX, Hong Kong HKEX, Australia ASRS, New Zealand XRB
TCFD Pillars: Governance (2), Strategy (3), Risk Management (3), Metrics & Targets (3) = 11 disclosures
"""

__version__ = "1.0.0"
__standard__ = "TCFD (June 2017) + IFRS S2 (June 2023)"

from .config import TCFDAppConfig
from .disclosure_generator import DisclosureGenerator
from .issb_crosswalk_engine import ISSBCrosswalkEngine
from .gap_analysis_engine import GapAnalysisEngine
from .recommendation_engine import RecommendationEngine
from .data_quality_engine import DataQualityEngine
from .setup import TCFDPlatform, create_app

__all__ = [
    "__version__",
    "__standard__",
    # Configuration
    "TCFDAppConfig",
    # Service Engines
    "DisclosureGenerator",
    "ISSBCrosswalkEngine",
    "GapAnalysisEngine",
    "RecommendationEngine",
    "DataQualityEngine",
    # Platform Facade
    "TCFDPlatform",
    "create_app",
]
