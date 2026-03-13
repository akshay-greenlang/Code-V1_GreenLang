# -*- coding: utf-8 -*-
"""
Corruption Index Monitor Agent - AGENT-EUDR-019

Production-grade corruption index monitoring platform for EUDR compliance
covering Transparency International Corruption Perceptions Index (CPI)
monitoring with 180+ country scores on a 0-100 scale (0=most corrupt,
100=cleanest); World Bank Worldwide Governance Indicators (WGI) analysis
across 6 dimensions (voice_accountability, political_stability,
government_effectiveness, regulatory_quality, rule_of_law,
control_of_corruption) on a -2.5 to +2.5 scale; sector-specific bribery
risk assessment for forestry, customs, agriculture, mining, extraction,
and judiciary with configurable weights; institutional quality scoring
incorporating judicial independence, regulatory enforcement, forest
governance, and law enforcement capacity; trend analysis with minimum
5-year trajectories, 10-year windows, and 3-year prediction horizons;
deforestation-corruption correlation analysis using Pearson correlation
with configurable significance levels; alert generation for CPI changes
exceeding 5 points, WGI changes exceeding 0.3 units, and trend
reversals; and compliance impact assessment mapping corruption indices
to EUDR Article 29 country classifications (low/standard/high risk)
with due diligence level determination (simplified/standard/enhanced).

This package provides a complete corruption index monitoring system for
EUDR regulatory compliance per EU 2023/1115 Articles 10, 11, 13, 29,
and 31:

    Capabilities:
        - CPI score monitoring for 180+ countries with percentile ranking,
          regional aggregation, year-over-year change detection, and
          Transparency International data source integration
        - WGI analysis across 6 governance dimensions with standard error
          tracking, percentile rank computation, composite governance
          scoring, and World Bank data source integration
        - Sector-specific bribery risk assessment for 6 sectors (forestry,
          customs, agriculture, mining, extraction, judiciary) with
          contributing factor identification, mitigation measure
          recommendations, and configurable sector weights
        - Institutional quality scoring combining judicial independence,
          regulatory enforcement effectiveness, forest governance
          quality, and law enforcement capacity into composite scores
        - Trend analysis with linear regression, R-squared goodness of
          fit, trajectory prediction, confidence intervals, minimum
          5-year data requirements, and trend reversal detection
        - Deforestation-corruption correlation using Pearson correlation
          coefficient, p-value significance testing, regression model
          parameters, and minimum 10 data point requirements
        - Alert generation for significant index changes, trend reversals,
          threshold breaches, and country reclassification events with
          configurable severity levels (low, medium, high, critical)
        - Compliance impact assessment mapping CPI/WGI scores to EUDR
          Article 29 country classifications with due diligence level
          determination and risk adjustment factors

    Foundational modules:
        - config: CorruptionIndexMonitorConfig with GL_EUDR_CIM_
          env var support (90+ settings)
        - models: Pydantic v2 data models with 10 enumerations,
          10 core models, 10 request models, and 10 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          10 entity types and 12 actions
        - metrics: 16 Prometheus self-monitoring metrics (gl_eudr_cim_)

PRD: PRD-AGENT-EUDR-019
Agent ID: GL-EUDR-CIM-019
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 13, 29, 31
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.corruption_index_monitor import (
    ...     CPIScore,
    ...     WGIDimension,
    ...     RiskLevel,
    ...     CountryClassification,
    ...     CorruptionIndexMonitorConfig,
    ...     get_config,
    ... )
    >>> from decimal import Decimal
    >>> score = CPIScore(
    ...     country_code="BR",
    ...     year=2024,
    ...     score=Decimal("38"),
    ...     rank=104,
    ...     percentile=Decimal("42.2"),
    ...     region="americas",
    ...     data_source="transparency_international",
    ... )
    >>> cfg = get_config()
    >>> print(cfg.cpi_high_risk_threshold, cfg.wgi_risk_threshold)
    30 -0.5

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-CIM-019"

# ---------------------------------------------------------------------------
# Foundational imports: config
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.config import (
        CorruptionIndexMonitorConfig,
        get_config,
        reset_config,
        set_config,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import config module: {e}")
    CorruptionIndexMonitorConfig = None  # type: ignore[misc,assignment]
    get_config = None  # type: ignore[misc,assignment]
    reset_config = None  # type: ignore[misc,assignment]
    set_config = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: models
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.models import (
        # Enumerations (10)
        WGIDimension,
        RiskLevel,
        TrendDirection,
        AlertSeverity,
        ComplianceLevel,
        BriberySector,
        CountryClassification,
        CorrelationStrength,
        GovernanceRating,
        DataSource,
        # Core Models (10)
        CPIScore,
        WGIIndicator,
        BriberyRiskAssessment,
        InstitutionalQualityScore,
        TrendAnalysis,
        DeforestationCorrelation,
        Alert,
        ComplianceImpact,
        CountryProfile,
        AuditLogEntry,
        # Request Models (10)
        QueryCPIRequest,
        QueryWGIRequest,
        AssessBriberyRiskRequest,
        EvaluateInstitutionalQualityRequest,
        AnalyzeTrendRequest,
        AnalyzeCorrelationRequest,
        GenerateAlertRequest,
        AssessComplianceImpactRequest,
        BuildCountryProfileRequest,
        HealthCheckRequest,
        # Response Models (10)
        CPIScoreResponse,
        WGIIndicatorResponse,
        BriberyRiskResponse,
        InstitutionalQualityResponse,
        TrendAnalysisResponse,
        CorrelationResponse,
        AlertResponse,
        ComplianceImpactResponse,
        CountryProfileResponse,
        HealthCheckResponse,
        # Constants
        VERSION,
        EUDR_CUTOFF_DATE,
        MAX_CPI_SCORE,
        MIN_CPI_SCORE,
        MAX_WGI_ESTIMATE,
        MIN_WGI_ESTIMATE,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        SUPPORTED_REGIONS,
        WGI_DIMENSIONS_LIST,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import models module: {e}")
    # Enumerations (10)
    WGIDimension = None  # type: ignore[misc,assignment]
    RiskLevel = None  # type: ignore[misc,assignment]
    TrendDirection = None  # type: ignore[misc,assignment]
    AlertSeverity = None  # type: ignore[misc,assignment]
    ComplianceLevel = None  # type: ignore[misc,assignment]
    BriberySector = None  # type: ignore[misc,assignment]
    CountryClassification = None  # type: ignore[misc,assignment]
    CorrelationStrength = None  # type: ignore[misc,assignment]
    GovernanceRating = None  # type: ignore[misc,assignment]
    DataSource = None  # type: ignore[misc,assignment]
    # Core Models (10)
    CPIScore = None  # type: ignore[misc,assignment]
    WGIIndicator = None  # type: ignore[misc,assignment]
    BriberyRiskAssessment = None  # type: ignore[misc,assignment]
    InstitutionalQualityScore = None  # type: ignore[misc,assignment]
    TrendAnalysis = None  # type: ignore[misc,assignment]
    DeforestationCorrelation = None  # type: ignore[misc,assignment]
    Alert = None  # type: ignore[misc,assignment]
    ComplianceImpact = None  # type: ignore[misc,assignment]
    CountryProfile = None  # type: ignore[misc,assignment]
    AuditLogEntry = None  # type: ignore[misc,assignment]
    # Request Models (10)
    QueryCPIRequest = None  # type: ignore[misc,assignment]
    QueryWGIRequest = None  # type: ignore[misc,assignment]
    AssessBriberyRiskRequest = None  # type: ignore[misc,assignment]
    EvaluateInstitutionalQualityRequest = None  # type: ignore[misc,assignment]
    AnalyzeTrendRequest = None  # type: ignore[misc,assignment]
    AnalyzeCorrelationRequest = None  # type: ignore[misc,assignment]
    GenerateAlertRequest = None  # type: ignore[misc,assignment]
    AssessComplianceImpactRequest = None  # type: ignore[misc,assignment]
    BuildCountryProfileRequest = None  # type: ignore[misc,assignment]
    HealthCheckRequest = None  # type: ignore[misc,assignment]
    # Response Models (10)
    CPIScoreResponse = None  # type: ignore[misc,assignment]
    WGIIndicatorResponse = None  # type: ignore[misc,assignment]
    BriberyRiskResponse = None  # type: ignore[misc,assignment]
    InstitutionalQualityResponse = None  # type: ignore[misc,assignment]
    TrendAnalysisResponse = None  # type: ignore[misc,assignment]
    CorrelationResponse = None  # type: ignore[misc,assignment]
    AlertResponse = None  # type: ignore[misc,assignment]
    ComplianceImpactResponse = None  # type: ignore[misc,assignment]
    CountryProfileResponse = None  # type: ignore[misc,assignment]
    HealthCheckResponse = None  # type: ignore[misc,assignment]
    # Constants
    VERSION = None  # type: ignore[misc,assignment]
    EUDR_CUTOFF_DATE = None  # type: ignore[misc,assignment]
    MAX_CPI_SCORE = None  # type: ignore[misc,assignment]
    MIN_CPI_SCORE = None  # type: ignore[misc,assignment]
    MAX_WGI_ESTIMATE = None  # type: ignore[misc,assignment]
    MIN_WGI_ESTIMATE = None  # type: ignore[misc,assignment]
    MAX_BATCH_SIZE = None  # type: ignore[misc,assignment]
    EUDR_RETENTION_YEARS = None  # type: ignore[misc,assignment]
    SUPPORTED_REGIONS = None  # type: ignore[misc,assignment]
    WGI_DIMENSIONS_LIST = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_tracker,
        reset_tracker,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import provenance module: {e}")
    ProvenanceRecord = None  # type: ignore[misc,assignment]
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_tracker = None  # type: ignore[misc,assignment]
    reset_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Foundational imports: metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.metrics import (
        PROMETHEUS_AVAILABLE,
        # Counter helpers
        record_cpi_query,
        record_wgi_query,
        record_bribery_assessment,
        record_institutional_assessment,
        record_trend_analysis,
        record_correlation_analysis,
        record_alert_generated,
        record_compliance_impact,
        record_api_error,
        # Histogram helpers
        observe_query_duration,
        observe_analysis_duration,
        observe_correlation_duration,
        # Gauge helpers
        set_monitored_countries,
        set_high_risk_countries,
        set_active_alerts,
        set_data_freshness_days,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import metrics module: {e}")
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    record_cpi_query = None  # type: ignore[misc,assignment]
    record_wgi_query = None  # type: ignore[misc,assignment]
    record_bribery_assessment = None  # type: ignore[misc,assignment]
    record_institutional_assessment = None  # type: ignore[misc,assignment]
    record_trend_analysis = None  # type: ignore[misc,assignment]
    record_correlation_analysis = None  # type: ignore[misc,assignment]
    record_alert_generated = None  # type: ignore[misc,assignment]
    record_compliance_impact = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]
    observe_query_duration = None  # type: ignore[misc,assignment]
    observe_analysis_duration = None  # type: ignore[misc,assignment]
    observe_correlation_duration = None  # type: ignore[misc,assignment]
    set_monitored_countries = None  # type: ignore[misc,assignment]
    set_high_risk_countries = None  # type: ignore[misc,assignment]
    set_active_alerts = None  # type: ignore[misc,assignment]
    set_data_freshness_days = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional - engines may not exist yet)
# ---------------------------------------------------------------------------

# ---- Engine 1: CPI Monitor Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.cpi_monitor_engine import (
        CPIMonitorEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.cpi_monitor_engine import (
            CPIMonitorEngine,
        )
    except ImportError:
        CPIMonitorEngine = None  # type: ignore[misc,assignment]

# ---- Engine 2: WGI Analyzer Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.wgi_analyzer_engine import (
        WGIAnalyzerEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.wgi_analyzer_engine import (
            WGIAnalyzerEngine,
        )
    except ImportError:
        WGIAnalyzerEngine = None  # type: ignore[misc,assignment]

# ---- Engine 3: Bribery Risk Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.bribery_risk_engine import (
        BriberyRiskEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.bribery_risk_engine import (
            BriberyRiskEngine,
        )
    except ImportError:
        BriberyRiskEngine = None  # type: ignore[misc,assignment]

# ---- Engine 4: Institutional Quality Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.institutional_quality_engine import (
        InstitutionalQualityEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.institutional_quality_engine import (
            InstitutionalQualityEngine,
        )
    except ImportError:
        InstitutionalQualityEngine = None  # type: ignore[misc,assignment]

# ---- Engine 5: Trend Analysis Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.trend_analysis_engine import (
        TrendAnalysisEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.trend_analysis_engine import (
            TrendAnalysisEngine,
        )
    except ImportError:
        TrendAnalysisEngine = None  # type: ignore[misc,assignment]

# ---- Engine 6: Deforestation Correlation Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.deforestation_correlation_engine import (
        DeforestationCorrelationEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.deforestation_correlation_engine import (
            DeforestationCorrelationEngine,
        )
    except ImportError:
        DeforestationCorrelationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Alert Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.alert_engine import (
        AlertEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.alert_engine import (
            AlertEngine,
        )
    except ImportError:
        AlertEngine = None  # type: ignore[misc,assignment]

# ---- Engine 8: Compliance Impact Engine ----
try:
    from greenlang.agents.eudr.corruption_index_monitor.compliance_impact_engine import (
        ComplianceImpactEngine,
    )
except ImportError:
    try:
        from greenlang.agents.eudr.corruption_index_monitor.engines.compliance_impact_engine import (
            ComplianceImpactEngine,
        )
    except ImportError:
        ComplianceImpactEngine = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Service facade import (conditional - service may not exist yet)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.service import (
        CorruptionIndexMonitorService,
    )
except ImportError:
    CorruptionIndexMonitorService = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Config --
    "CorruptionIndexMonitorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Enumerations (10) --
    "WGIDimension",
    "RiskLevel",
    "TrendDirection",
    "AlertSeverity",
    "ComplianceLevel",
    "BriberySector",
    "CountryClassification",
    "CorrelationStrength",
    "GovernanceRating",
    "DataSource",
    # -- Core Models (10) --
    "CPIScore",
    "WGIIndicator",
    "BriberyRiskAssessment",
    "InstitutionalQualityScore",
    "TrendAnalysis",
    "DeforestationCorrelation",
    "Alert",
    "ComplianceImpact",
    "CountryProfile",
    "AuditLogEntry",
    # -- Request Models (10) --
    "QueryCPIRequest",
    "QueryWGIRequest",
    "AssessBriberyRiskRequest",
    "EvaluateInstitutionalQualityRequest",
    "AnalyzeTrendRequest",
    "AnalyzeCorrelationRequest",
    "GenerateAlertRequest",
    "AssessComplianceImpactRequest",
    "BuildCountryProfileRequest",
    "HealthCheckRequest",
    # -- Response Models (10) --
    "CPIScoreResponse",
    "WGIIndicatorResponse",
    "BriberyRiskResponse",
    "InstitutionalQualityResponse",
    "TrendAnalysisResponse",
    "CorrelationResponse",
    "AlertResponse",
    "ComplianceImpactResponse",
    "CountryProfileResponse",
    "HealthCheckResponse",
    # -- Constants --
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_CPI_SCORE",
    "MIN_CPI_SCORE",
    "MAX_WGI_ESTIMATE",
    "MIN_WGI_ESTIMATE",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "SUPPORTED_REGIONS",
    "WGI_DIMENSIONS_LIST",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_tracker",
    "reset_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_cpi_query",
    "record_wgi_query",
    "record_bribery_assessment",
    "record_institutional_assessment",
    "record_trend_analysis",
    "record_correlation_analysis",
    "record_alert_generated",
    "record_compliance_impact",
    "record_api_error",
    "observe_query_duration",
    "observe_analysis_duration",
    "observe_correlation_duration",
    "set_monitored_countries",
    "set_high_risk_countries",
    "set_active_alerts",
    "set_data_freshness_days",
    # -- Engines (8) --
    "CPIMonitorEngine",
    "WGIAnalyzerEngine",
    "BriberyRiskEngine",
    "InstitutionalQualityEngine",
    "TrendAnalysisEngine",
    "DeforestationCorrelationEngine",
    "AlertEngine",
    "ComplianceImpactEngine",
    # -- Service Facade --
    "CorruptionIndexMonitorService",
]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, and model counts for the Corruption Index
        Monitor agent.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-CIM-019'
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Corruption Index Monitor",
        "prd": "PRD-AGENT-EUDR-019",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["10", "11", "13", "29", "31"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "data_sources": [
            "Transparency International CPI",
            "World Bank WGI",
            "TRACE Bribery Risk Matrix",
            "Global Forest Watch",
        ],
        "wgi_dimensions": [
            "voice_accountability",
            "political_stability",
            "government_effectiveness",
            "regulatory_quality",
            "rule_of_law",
            "control_of_corruption",
        ],
        "bribery_sectors": [
            "forestry",
            "customs",
            "agriculture",
            "mining",
            "extraction",
            "judiciary",
        ],
        "engines": [
            "CPIMonitorEngine",
            "WGIAnalyzerEngine",
            "BriberyRiskEngine",
            "InstitutionalQualityEngine",
            "TrendAnalysisEngine",
            "DeforestationCorrelationEngine",
            "AlertEngine",
            "ComplianceImpactEngine",
        ],
        "engine_count": 8,
        "enum_count": 10,
        "core_model_count": 10,
        "request_model_count": 10,
        "response_model_count": 10,
        "metrics_count": 16,
        "db_prefix": "gl_eudr_cim_",
        "metrics_prefix": "gl_eudr_cim_",
        "env_prefix": "GL_EUDR_CIM_",
    }
