# -*- coding: utf-8 -*-
"""
PACK-038 Peak Shaving Pack - Engines Package
==============================================

Production-grade calculation engines for peak shaving, demand charge
management, BESS sizing, load shifting, coincident peak management,
ratchet analysis, power factor correction, financial modelling, and
comprehensive reporting.

Engines:
    1. LoadProfileEngine        - 15-minute interval load profile analysis
    2. PeakIdentifierEngine     - Peak detection and attribution
    3. DemandChargeEngine       - Tariff decomposition and demand charge calc
    4. BESSSizingEngine         - Battery storage sizing optimisation
    5. LoadShiftingEngine       - Load shifting with constraint satisfaction
    6. CPManagementEngine       - Coincident peak (CP) management
    7. RatchetAnalysisEngine    - Ratchet demand persistence analysis
    8. PowerFactorEngine        - Power factor and reactive power analysis
    9. FinancialEngine          - Investment financial modelling
   10. PeakReportingEngine      - Dashboard panels & report generation

Architecture:
    All engines use deterministic calculations (no LLM in calc path),
    Pydantic v2 models, Python Decimal for financial precision, and
    SHA-256 provenance hashing for audit trail integrity.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-038 Peak Shaving
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack_id__: str = "PACK-038"
__pack_name__: str = "Peak Shaving Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Load Profile
# ---------------------------------------------------------------------------
try:
    from .load_profile_engine import (
        LoadProfileEngine,
    )
    _loaded_engines.append("LoadProfileEngine")
except ImportError as e:
    logger.debug("Engine 1 (LoadProfileEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 2: Peak Identifier
# ---------------------------------------------------------------------------
try:
    from .peak_identifier_engine import (
        PeakIdentifierEngine,
    )
    _loaded_engines.append("PeakIdentifierEngine")
except ImportError as e:
    logger.debug("Engine 2 (PeakIdentifierEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 3: Demand Charge
# ---------------------------------------------------------------------------
try:
    from .demand_charge_engine import (
        DemandChargeEngine,
    )
    _loaded_engines.append("DemandChargeEngine")
except ImportError as e:
    logger.debug("Engine 3 (DemandChargeEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 4: BESS Sizing
# ---------------------------------------------------------------------------
try:
    from .bess_sizing_engine import (
        BESSSizingEngine,
    )
    _loaded_engines.append("BESSSizingEngine")
except ImportError as e:
    logger.debug("Engine 4 (BESSSizingEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 5: Load Shifting
# ---------------------------------------------------------------------------
try:
    from .load_shifting_engine import (
        LoadShiftingEngine,
    )
    _loaded_engines.append("LoadShiftingEngine")
except ImportError as e:
    logger.debug("Engine 5 (LoadShiftingEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 6: CP Management
# ---------------------------------------------------------------------------
try:
    from .cp_management_engine import (
        AlertLevel,
        CPCharge,
        CPEvent,
        CPManagementEngine,
        CPManagementResult,
        CPMethodology,
        CPPrediction,
        CPResponse,
        CPStatus,
        PredictionConfidence,
        ResponseAction,
    )
    _loaded_engines.append("CPManagementEngine")
except ImportError as e:
    logger.debug("Engine 6 (CPManagementEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 7: Ratchet Analysis
# ---------------------------------------------------------------------------
try:
    from .ratchet_analysis_engine import (
        PreventionPlan,
        PreventionStrategy,
        RatchetAnalysisEngine,
        RatchetDemand,
        RatchetImpact,
        RatchetResult,
        RatchetType,
        RatchetPercentage,
        RiskLevel as RatchetRiskLevel,
        SpikeAnalysis,
        SpikeCause,
    )
    _loaded_engines.append("RatchetAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 7 (RatchetAnalysisEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 8: Power Factor
# ---------------------------------------------------------------------------
try:
    from .power_factor_engine import (
        BillingMethod,
        CorrectionSizing,
        CorrectionType,
        HarmonicOrder,
        HarmonicProfile,
        LoadCategory,
        PFStatus,
        PowerFactorEngine,
        PowerFactorReading,
        PowerFactorResult,
        ReactiveAnalysis,
    )
    _loaded_engines.append("PowerFactorEngine")
except ImportError as e:
    logger.debug("Engine 8 (PowerFactorEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 9: Financial
# ---------------------------------------------------------------------------
try:
    from .financial_engine import (
        CashFlowProjection,
        FinancialEngine,
        FinancialMetric,
        FinancialResult,
        IncentiveCapture,
        IncentiveProgram,
        InvestmentCase,
        InvestmentType,
        RevenueStack,
        RiskLevel as FinancialRiskLevel,
        ScenarioType,
    )
    _loaded_engines.append("FinancialEngine")
except ImportError as e:
    logger.debug("Engine 9 (FinancialEngine) not available: %s", e)

# ---------------------------------------------------------------------------
# Engine 10: Peak Reporting
# ---------------------------------------------------------------------------
try:
    from .peak_reporting_engine import (
        DashboardData,
        DashboardPanel,
        DashboardWidget,
        ExportFormat,
        KPIMetric,
        PeakReportingEngine,
        ReportOutput,
        ReportSection,
        ReportType,
        TrendDirection,
        WidgetType,
    )
    _loaded_engines.append("PeakReportingEngine")
except ImportError as e:
    logger.debug("Engine 10 (PeakReportingEngine) not available: %s", e)


__all__: list[str] = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "__engines_count__",
    # --- Engine 1: Load Profile ---
    "LoadProfileEngine",
    # --- Engine 2: Peak Identifier ---
    "PeakIdentifierEngine",
    # --- Engine 3: Demand Charge ---
    "DemandChargeEngine",
    # --- Engine 4: BESS Sizing ---
    "BESSSizingEngine",
    # --- Engine 5: Load Shifting ---
    "LoadShiftingEngine",
    # --- Engine 6: CP Management ---
    "CPManagementEngine",
    "CPEvent",
    "CPPrediction",
    "CPResponse",
    "CPCharge",
    "CPManagementResult",
    "CPMethodology",
    "CPStatus",
    "ResponseAction",
    "PredictionConfidence",
    "AlertLevel",
    # --- Engine 7: Ratchet Analysis ---
    "RatchetAnalysisEngine",
    "RatchetDemand",
    "SpikeAnalysis",
    "RatchetImpact",
    "PreventionPlan",
    "RatchetResult",
    "RatchetType",
    "RatchetPercentage",
    "SpikeCause",
    "PreventionStrategy",
    "RatchetRiskLevel",
    # --- Engine 8: Power Factor ---
    "PowerFactorEngine",
    "PowerFactorReading",
    "ReactiveAnalysis",
    "CorrectionSizing",
    "HarmonicProfile",
    "PowerFactorResult",
    "PFStatus",
    "CorrectionType",
    "LoadCategory",
    "HarmonicOrder",
    "BillingMethod",
    # --- Engine 9: Financial ---
    "FinancialEngine",
    "InvestmentCase",
    "IncentiveCapture",
    "RevenueStack",
    "CashFlowProjection",
    "FinancialResult",
    "InvestmentType",
    "IncentiveProgram",
    "FinancialMetric",
    "FinancialRiskLevel",
    "ScenarioType",
    # --- Engine 10: Peak Reporting ---
    "PeakReportingEngine",
    "KPIMetric",
    "DashboardWidget",
    "ReportSection",
    "DashboardData",
    "ReportOutput",
    "DashboardPanel",
    "ReportType",
    "ExportFormat",
    "WidgetType",
    "TrendDirection",
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


logger.info(
    "PACK-038 Peak Shaving engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
