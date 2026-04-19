"""
GL-020 ECONOPULSE REST API

FastAPI-based REST API for Economizer Performance Monitoring.
Provides endpoints for performance tracking, fouling analysis, alerts,
efficiency metrics, soot blower integration, and reporting.

Agent ID: GL-020
Codename: ECONOPULSE
Name: EconomizerPerformanceAgent
Description: Monitors economizer performance and fouling
"""

from .main import app
from .routes import router
from .schemas import (
    # Economizer models
    Economizer,
    EconomizerList,
    EconomizerCreate,
    EconomizerUpdate,
    # Performance models
    PerformanceMetrics,
    PerformanceHistory,
    PerformanceTrend,
    # Fouling models
    FoulingStatus,
    FoulingHistory,
    FoulingPrediction,
    FoulingBaseline,
    # Alert models
    Alert,
    AlertList,
    AlertAcknowledge,
    AlertThreshold,
    AlertThresholdConfig,
    # Efficiency models
    EfficiencyMetrics,
    EfficiencyLoss,
    EfficiencySavings,
    # Soot blower models
    SootBlowerStatus,
    SootBlowerTrigger,
    CleaningHistory,
    CleaningOptimization,
    # Report models
    DailyReport,
    WeeklyReport,
    EfficiencyReport,
    ReportExport,
    # Health models
    HealthStatus,
    ReadinessStatus,
)

__version__ = "1.0.0"
__agent_id__ = "GL-020"
__codename__ = "ECONOPULSE"

__all__ = [
    "app",
    "router",
    "__version__",
    "__agent_id__",
    "__codename__",
]
