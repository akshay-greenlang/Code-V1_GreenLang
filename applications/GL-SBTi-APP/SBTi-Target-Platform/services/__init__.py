"""
GL-SBTi-APP v1.0 -- SBTi Target Validation Platform Services

This package provides configuration, domain models, and 14 service engines plus
setup facade for implementing the SBTi (Science Based Targets initiative)
target-setting, validation, and progress-tracking framework covering Corporate
V5.3 near-term/long-term targets, Net-Zero V1.3, Financial Institutions (FINZ
V1.0), and FLAG (Forest, Land and Agriculture) guidance.

Engines (14 service engines + config + models + setup):
    config: Enumerations, pathway parameters, sector benchmarks, and app settings.
    models: Pydantic v2 domain models for all SBTi entities.
    target_setting_engine: Near-term, long-term, and net-zero target formulation.
    pathway_engine: ACA, SDA, and FLAG decarbonization pathway computation.
    validation_engine: 42-criterion automated target validation (C1-C28 + NZ).
    scope3_screening_engine: Scope 3 materiality screening and category coverage.
    flag_engine: FLAG commodity deforestation and land-use target assessment.
    sector_engine: SDA sector intensity benchmarks and pathway alignment.
    progress_tracking_engine: Annual progress, variance analysis, and recalculation.
    temperature_scoring_engine: Portfolio and target temperature rating (ITR).
    recalculation_engine: Significant-change triggers and base-year recalculation.
    review_engine: SBTi submission workflow, status tracking, and appeal management.
    fi_engine: Financial institution portfolio target setting (FINZ V1.0).
    framework_crosswalk_engine: Cross-framework alignment (CDP, TCFD, CSRD, GHG).
    setup: Platform facade composing all engines with FastAPI app factory.

Standard: SBTi Corporate Manual V5.3 (2025)
Net-Zero: SBTi Corporate Net-Zero Standard V1.3 (2025)
Financial Institutions: SBTi FINZ V1.0 (2024)
FLAG: SBTi Forest, Land and Agriculture Guidance V1.1 (2024)
Pathways: ACA (cross-sector), SDA (sector-specific), FLAG (land sector)
Target Types: Near-term (5-10yr), Long-term (by 2050), Net-zero
Scope Coverage: Scope 1+2 (>=95%), Scope 3 (>=67% near-term, >=90% long-term)
Sectors: 12 SDA sectors with intensity metrics
FLAG Commodities: 11 commodity categories (cattle, soy, palm oil, etc.)
"""

__version__ = "1.0.0"
__standard__ = "SBTi Corporate V5.3 + Net-Zero V1.3 + FINZ V1.0 + FLAG"

from .config import SBTiAppConfig
from .target_setting_engine import TargetSettingEngine
from .pathway_engine import PathwayEngine
from .validation_engine import ValidationEngine
from .scope3_screening_engine import Scope3ScreeningEngine
from .flag_engine import FLAGEngine
from .sector_engine import SectorEngine
from .progress_tracking_engine import ProgressTrackingEngine
from .temperature_scoring_engine import TemperatureScoringEngine
from .recalculation_engine import RecalculationEngine
from .review_engine import ReviewEngine
from .fi_engine import FIEngine
from .framework_crosswalk_engine import FrameworkCrosswalkEngine
from .data_quality_engine import DataQualityEngine
from .setup import SBTiPlatform, create_app

__all__ = [
    "__version__",
    "__standard__",
    # Configuration
    "SBTiAppConfig",
    # Service Engines (14)
    "TargetSettingEngine",
    "PathwayEngine",
    "ValidationEngine",
    "Scope3ScreeningEngine",
    "FLAGEngine",
    "SectorEngine",
    "ProgressTrackingEngine",
    "TemperatureScoringEngine",
    "RecalculationEngine",
    "ReviewEngine",
    "FIEngine",
    "FrameworkCrosswalkEngine",
    "DataQualityEngine",
    # Platform Facade
    "SBTiPlatform",
    "create_app",
]
