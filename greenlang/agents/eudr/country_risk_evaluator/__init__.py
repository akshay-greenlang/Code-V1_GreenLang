# -*- coding: utf-8 -*-
"""
Country Risk Evaluator Agent - AGENT-EUDR-016

Production-grade country risk evaluation platform for EUDR compliance
covering composite country risk scoring with 6 weighted factors
(deforestation rate 30%, governance index 20%, enforcement score 15%,
corruption perception 15%, forest law compliance 10%, historical
trend 10%); commodity-specific risk analysis for all 7 EUDR-regulated
commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood);
sub-national deforestation hotspot detection with DBSCAN spatial
clustering, FIRMS/VIIRS fire correlation, protected area proximity,
and indigenous territory overlap detection; governance index engine
integrating World Bank WGI (6 dimensions), Transparency International
CPI, and FAO/ITTO forest governance frameworks; automated 3-tier
due diligence classification per EUDR Articles 10-13 (simplified,
standard, enhanced) with certification credits, cost estimation, and
audit frequency recommendation; bilateral trade flow analysis with
re-export risk detection, commodity laundering identification, and
HS code mapping; audit-ready risk report generation in PDF/JSON/HTML
with multi-language support (EN/FR/DE/ES/PT); and EC regulatory update
tracking with country reclassification impact assessment and
enforcement action monitoring.

This package provides a complete country risk evaluation system for
EUDR regulatory compliance per EU 2023/1115 Articles 10, 11, 13,
29, and 31:

    Capabilities:
        - Composite country risk scoring for 200+ countries using
          6-factor weighted formula aligned with EC Article 29
          benchmarking methodology, deterministic Decimal arithmetic,
          configurable weights (5-50% per factor, sum 100%), and
          EC benchmark validation with critical alert on mismatch
        - Commodity-specific risk analysis for all 7 EUDR commodities
          per country, incorporating production volume, deforestation
          correlation, certification effectiveness (FSC, RSPO, RA,
          Fairtrade, PEFC, Bonsucro, ISCC, Organic), seasonal risk
          variation, and supply chain complexity scoring (1-10)
        - Sub-national deforestation hotspot detection using GFW tree
          cover loss data, DBSCAN clustering (min 10 alerts, 5km
          radius), FIRMS/VIIRS fire alert correlation, protected area
          proximity scoring, indigenous territory overlap detection,
          and linear regression trend analysis (5-year window)
        - Governance index engine with WGI (6 dimensions), CPI,
          FAO/ITTO forest governance, legal framework strength (5
          criteria), enforcement effectiveness, cross-border
          cooperation, indigenous rights, EIA compliance, judicial
          independence, and government transparency scoring
        - Automated 3-tier due diligence classification (simplified
          0-30, standard 31-65, enhanced 66-100) with dynamic
          reclassification, certification credits (max 30 points),
          cost estimation (EUR 200-15,000), audit frequency
          recommendation, and regulatory submission mapping
        - Trade flow analysis with bilateral mapping (200+ origins
          to 27 EU states), re-export risk detection, HS/CN code
          mapping (1,200+ codes), concentration risk (HHI), sanction
          overlay, FTA impact analysis, and port-of-entry profiling
        - Risk report generation in PDF/JSON/HTML/CSV/Excel with
          country profiles, commodity matrices, executive summaries,
          comparative analyses, trend reports, and DDS-ready
          documentation in 5 languages
        - Regulatory update tracking with EC benchmarking list
          monitoring, reclassification impact assessment, enforcement
          action tracking, grace period management, and stakeholder
          notification

    Foundational modules:
        - config: CountryRiskEvaluatorConfig with GL_EUDR_CRE_
          env var support (70+ settings)
        - models: Pydantic v2 data models with 15 enumerations,
          12 core models, 15 request models, and 15 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 12 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_cre_)

PRD: PRD-AGENT-EUDR-016
Agent ID: GL-EUDR-CRE-016
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 13, 29, 31
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.country_risk_evaluator import (
    ...     CountryRiskAssessment,
    ...     RiskLevel,
    ...     CommodityType,
    ...     CountryRiskEvaluatorConfig,
    ...     get_config,
    ... )
    >>> assessment = CountryRiskAssessment(
    ...     country_code="BR",
    ...     country_name="Brazil",
    ...     risk_level=RiskLevel.HIGH,
    ...     risk_score=72.5,
    ...     composite_factors={
    ...         "deforestation_rate": 85.0,
    ...         "governance_index": 55.0,
    ...         "enforcement_score": 45.0,
    ...         "corruption_index": 62.0,
    ...         "forest_law_compliance": 50.0,
    ...         "historical_trend": 70.0,
    ...     },
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-CRE-016"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.config import (
        CountryRiskEvaluatorConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    CountryRiskEvaluatorConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.models import (
        # Constants
        VERSION,
        EUDR_CUTOFF_DATE,
        MAX_RISK_SCORE,
        MIN_RISK_SCORE,
        MAX_BATCH_SIZE,
        EUDR_RETENTION_YEARS,
        EC_BENCHMARK_URL,
        EUDR_ENFORCEMENT_DATE,
        EUDR_SME_ENFORCEMENT_DATE,
        SUPPORTED_COMMODITIES,
        SUPPORTED_OUTPUT_FORMATS,
        SUPPORTED_REPORT_LANGUAGES,
        DEFAULT_FACTOR_WEIGHTS,
        SUPPORTED_COUNTRIES,
        # Enumerations
        RiskLevel,
        DueDiligenceLevel,
        CommodityType,
        ForestType,
        GovernanceIndicator,
        HotspotSeverity,
        DeforestationDriver,
        TradeFlowDirection,
        ReportFormat,
        ReportType,
        RegulatoryStatus,
        AssessmentConfidence,
        TrendDirection,
        CertificationScheme,
        DataSource,
        # Core Models
        CountryRiskAssessment,
        CommodityRiskProfile,
        DeforestationHotspot,
        GovernanceIndex,
        DueDiligenceClassification,
        TradeFlow,
        RiskReport,
        RegulatoryUpdate,
        RiskFactor,
        RiskHistory,
        CertificationRecord,
        AuditLogEntry,
        # Request Models
        AssessCountryRequest,
        AnalyzeCommodityRequest,
        DetectHotspotsRequest,
        EvaluateGovernanceRequest,
        ClassifyDueDiligenceRequest,
        AnalyzeTradeFlowRequest,
        GenerateReportRequest,
        TrackRegulatoryRequest,
        CompareCountriesRequest,
        GetTrendsRequest,
        CostEstimateRequest,
        MatrixRequest,
        ClusteringRequest,
        ImpactAssessmentRequest,
        SearchRequest,
        # Response Models
        CountryRiskResponse,
        CommodityRiskResponse,
        HotspotResponse,
        GovernanceResponse,
        DueDiligenceResponse,
        TradeFlowResponse,
        ReportResponse,
        RegulatoryResponse,
        ComparisonResponse,
        TrendResponse,
        CostEstimateResponse,
        MatrixResponse,
        ClusteringResponse,
        ImpactResponse,
        HealthResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        cre_assessments_total,
        cre_commodity_analyses_total,
        cre_hotspots_detected_total,
        cre_classifications_total,
        cre_reports_generated_total,
        cre_trade_analyses_total,
        cre_regulatory_updates_total,
        cre_api_errors_total,
        cre_assessment_duration_seconds,
        cre_commodity_analysis_duration_seconds,
        cre_hotspot_detection_duration_seconds,
        cre_classification_duration_seconds,
        cre_report_generation_duration_seconds,
        cre_active_hotspots,
        cre_countries_assessed,
        cre_high_risk_countries,
        cre_pending_reclassifications,
        cre_stale_assessments,
        # Helper functions
        record_assessment_completed,
        record_commodity_analysis,
        record_hotspot_detected,
        record_classification_completed,
        record_report_generated,
        record_trade_analysis,
        record_regulatory_update,
        record_api_error,
        observe_assessment_duration,
        observe_commodity_analysis_duration,
        observe_hotspot_detection_duration,
        observe_classification_duration,
        observe_report_generation_duration,
        set_active_hotspots,
        set_countries_assessed,
        set_high_risk_countries,
        set_pending_reclassifications,
        set_stale_assessments,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---- Engine 1: Country Risk Scorer ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.country_risk_scorer import CountryRiskScorer
except ImportError:
    CountryRiskScorer = None  # type: ignore[assignment,misc]

# ---- Engine 2: Commodity Risk Analyzer ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.commodity_risk_analyzer import CommodityRiskAnalyzer
except ImportError:
    CommodityRiskAnalyzer = None  # type: ignore[assignment,misc]

# ---- Engine 3: Deforestation Hotspot Detector ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.deforestation_hotspot_detector import DeforestationHotspotDetector
except ImportError:
    DeforestationHotspotDetector = None  # type: ignore[assignment,misc]

# ---- Engine 4: Governance Index Engine ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.governance_index_engine import GovernanceIndexEngine
except ImportError:
    GovernanceIndexEngine = None  # type: ignore[assignment,misc]

# ---- Engine 5: Due Diligence Classifier ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.due_diligence_classifier import DueDiligenceClassifier
except ImportError:
    DueDiligenceClassifier = None  # type: ignore[assignment,misc]

# ---- Engine 6: Trade Flow Analyzer ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.trade_flow_analyzer import TradeFlowAnalyzer
except ImportError:
    TradeFlowAnalyzer = None  # type: ignore[assignment,misc]

# ---- Engine 7: Risk Report Generator ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.risk_report_generator import RiskReportGenerator
except ImportError:
    RiskReportGenerator = None  # type: ignore[assignment,misc]

# ---- Engine 8: Regulatory Update Tracker ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.regulatory_update_tracker import RegulatoryUpdateTracker
except ImportError:
    RegulatoryUpdateTracker = None  # type: ignore[assignment,misc]

# ---- Service Facade ----
try:
    from greenlang.agents.eudr.country_risk_evaluator.setup import CountryRiskEvaluatorService
except ImportError:
    CountryRiskEvaluatorService = None  # type: ignore[assignment,misc]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "CountryRiskEvaluatorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_CUTOFF_DATE",
    "MAX_RISK_SCORE",
    "MIN_RISK_SCORE",
    "MAX_BATCH_SIZE",
    "EUDR_RETENTION_YEARS",
    "EC_BENCHMARK_URL",
    "EUDR_ENFORCEMENT_DATE",
    "EUDR_SME_ENFORCEMENT_DATE",
    "SUPPORTED_COMMODITIES",
    "SUPPORTED_OUTPUT_FORMATS",
    "SUPPORTED_REPORT_LANGUAGES",
    "DEFAULT_FACTOR_WEIGHTS",
    "SUPPORTED_COUNTRIES",
    # -- Enumerations --
    "RiskLevel",
    "DueDiligenceLevel",
    "CommodityType",
    "ForestType",
    "GovernanceIndicator",
    "HotspotSeverity",
    "DeforestationDriver",
    "TradeFlowDirection",
    "ReportFormat",
    "ReportType",
    "RegulatoryStatus",
    "AssessmentConfidence",
    "TrendDirection",
    "CertificationScheme",
    "DataSource",
    # -- Core Models --
    "CountryRiskAssessment",
    "CommodityRiskProfile",
    "DeforestationHotspot",
    "GovernanceIndex",
    "DueDiligenceClassification",
    "TradeFlow",
    "RiskReport",
    "RegulatoryUpdate",
    "RiskFactor",
    "RiskHistory",
    "CertificationRecord",
    "AuditLogEntry",
    # -- Request Models --
    "AssessCountryRequest",
    "AnalyzeCommodityRequest",
    "DetectHotspotsRequest",
    "EvaluateGovernanceRequest",
    "ClassifyDueDiligenceRequest",
    "AnalyzeTradeFlowRequest",
    "GenerateReportRequest",
    "TrackRegulatoryRequest",
    "CompareCountriesRequest",
    "GetTrendsRequest",
    "CostEstimateRequest",
    "MatrixRequest",
    "ClusteringRequest",
    "ImpactAssessmentRequest",
    "SearchRequest",
    # -- Response Models --
    "CountryRiskResponse",
    "CommodityRiskResponse",
    "HotspotResponse",
    "GovernanceResponse",
    "DueDiligenceResponse",
    "TradeFlowResponse",
    "ReportResponse",
    "RegulatoryResponse",
    "ComparisonResponse",
    "TrendResponse",
    "CostEstimateResponse",
    "MatrixResponse",
    "ClusteringResponse",
    "ImpactResponse",
    "HealthResponse",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "cre_assessments_total",
    "cre_commodity_analyses_total",
    "cre_hotspots_detected_total",
    "cre_classifications_total",
    "cre_reports_generated_total",
    "cre_trade_analyses_total",
    "cre_regulatory_updates_total",
    "cre_api_errors_total",
    "cre_assessment_duration_seconds",
    "cre_commodity_analysis_duration_seconds",
    "cre_hotspot_detection_duration_seconds",
    "cre_classification_duration_seconds",
    "cre_report_generation_duration_seconds",
    "cre_active_hotspots",
    "cre_countries_assessed",
    "cre_high_risk_countries",
    "cre_pending_reclassifications",
    "cre_stale_assessments",
    "record_assessment_completed",
    "record_commodity_analysis",
    "record_hotspot_detected",
    "record_classification_completed",
    "record_report_generated",
    "record_trade_analysis",
    "record_regulatory_update",
    "record_api_error",
    "observe_assessment_duration",
    "observe_commodity_analysis_duration",
    "observe_hotspot_detection_duration",
    "observe_classification_duration",
    "observe_report_generation_duration",
    "set_active_hotspots",
    "set_countries_assessed",
    "set_high_risk_countries",
    "set_pending_reclassifications",
    "set_stale_assessments",
    # -- Engines --
    "CountryRiskScorer",
    "CommodityRiskAnalyzer",
    "DeforestationHotspotDetector",
    "GovernanceIndexEngine",
    "DueDiligenceClassifier",
    "TradeFlowAnalyzer",
    "RiskReportGenerator",
    "RegulatoryUpdateTracker",
    # -- Service Facade --
    "CountryRiskEvaluatorService",
]
