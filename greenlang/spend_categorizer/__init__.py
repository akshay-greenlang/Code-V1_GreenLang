# -*- coding: utf-8 -*-
"""
GL-DATA-SUP-002: GreenLang Spend Data Categorizer Agent SDK
============================================================

This package provides spend data ingestion, taxonomy classification,
GHG Protocol Scope 3 mapping, spend-based emission calculation,
rule-based categorization, analytics, reporting, and provenance
tracking SDK for the GreenLang framework. It supports:

- Multi-source spend record ingestion (CSV, Excel, API, ERP, manual)
- Multi-taxonomy classification (UNSPSC, NAICS, NACE, custom)
- Rule-based and keyword-based categorization with confidence scoring
- GHG Protocol Scope 3 category mapping (15 categories)
- Spend-based emission calculation (EPA EEIO, EXIOBASE, DEFRA)
- Emission factor lookup and management
- Hotspot analysis and trend analytics
- Report generation (JSON, CSV, Excel, PDF)
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_SPEND_CAT_ env prefix

Key Components:
    - config: SpendCategorizerConfig with GL_SPEND_CAT_ env prefix
    - record_ingestion: Spend record ingestion engine
    - taxonomy_classifier: Multi-taxonomy classification engine
    - scope3_mapper: GHG Protocol Scope 3 mapping engine
    - emission_calculator: Spend-based emission calculation engine
    - rule_engine: Rule-based categorization engine
    - spend_analytics: Analytics and hotspot engine
    - report_generator: Report generation engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: SpendCategorizerService facade

Example:
    >>> from greenlang.spend_categorizer import SpendCategorizerService
    >>> service = SpendCategorizerService()
    >>> records = service.ingest_records([
    ...     {"vendor_name": "Acme Corp", "amount": 50000, "description": "Office supplies"},
    ... ])
    >>> print(records[0].status)
    ingested

Agent ID: GL-DATA-SUP-002
Agent Name: Spend Data Categorizer Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-SUP-002"
__agent_name__ = "Spend Data Categorizer Agent"

# SDK availability flag
SPEND_CATEGORIZER_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.spend_categorizer.config import (
    SpendCategorizerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.spend_categorizer.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.spend_categorizer.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    spend_cat_records_ingested_total,
    spend_cat_records_classified_total,
    spend_cat_scope3_mapped_total,
    spend_cat_emissions_calculated_total,
    spend_cat_rules_evaluated_total,
    spend_cat_reports_generated_total,
    spend_cat_classification_confidence,
    spend_cat_processing_duration_seconds,
    spend_cat_active_batches,
    spend_cat_total_spend_usd,
    spend_cat_processing_errors_total,
    spend_cat_emission_factor_lookups_total,
    # Helper functions
    record_ingestion,
    record_classification,
    record_scope3_mapping,
    record_emission_calculation,
    record_rule_evaluation,
    record_report_generation,
    record_classification_confidence,
    record_processing_duration,
    update_active_batches,
    update_total_spend,
    record_processing_error,
    record_factor_lookup,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK)
# ---------------------------------------------------------------------------
from greenlang.spend_categorizer.spend_ingestion import SpendIngestionEngine as RecordIngestionEngine
from greenlang.spend_categorizer.taxonomy_classifier import TaxonomyClassifierEngine as TaxonomyClassifier
from greenlang.spend_categorizer.scope3_mapper import Scope3MapperEngine as Scope3Mapper
from greenlang.spend_categorizer.emission_factor import EmissionFactorEngine as EmissionCalculator
from greenlang.spend_categorizer.category_rule import CategoryRuleEngine as RuleEngine
from greenlang.spend_categorizer.spend_analytics import SpendAnalyticsEngine
from greenlang.spend_categorizer.reporting import ReportingEngine as ReportGeneratorEngine

# ---------------------------------------------------------------------------
# Models (Layer 2 SDK)
# ---------------------------------------------------------------------------
from greenlang.spend_categorizer.models import (
    # Enumerations
    TaxonomySystem,
    Scope3Category,
    ConfidenceLevel,
    RecordStatus,
    ClassificationMethod,
    EmissionFactorSource,
    ReportFormat,
    RuleConditionType,
    # Core models
    SpendRecord,
    TaxonomyCode,
    ClassificationResult,
    Scope3Assignment,
    EmissionFactor,
    EmissionResult,
    CategoryRule,
    # Request models
    IngestRecordsRequest,
    ClassifyRequest,
    CalculateEmissionsRequest,
)

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.spend_categorizer.setup import (
    SpendCategorizerService,
    configure_spend_categorizer,
    get_spend_categorizer,
    get_router,
    # Models
    SpendRecordResponse,
    ClassificationResponse,
    Scope3AssignmentResponse,
    EmissionCalculationResponse,
    CategoryRuleResponse,
    AnalyticsResponse,
    ReportResponse,
    SpendCategorizerStatisticsResponse,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "SPEND_CATEGORIZER_SDK_AVAILABLE",
    # Configuration
    "SpendCategorizerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "spend_cat_records_ingested_total",
    "spend_cat_records_classified_total",
    "spend_cat_scope3_mapped_total",
    "spend_cat_emissions_calculated_total",
    "spend_cat_rules_evaluated_total",
    "spend_cat_reports_generated_total",
    "spend_cat_classification_confidence",
    "spend_cat_processing_duration_seconds",
    "spend_cat_active_batches",
    "spend_cat_total_spend_usd",
    "spend_cat_processing_errors_total",
    "spend_cat_emission_factor_lookups_total",
    # Metric helper functions
    "record_ingestion",
    "record_classification",
    "record_scope3_mapping",
    "record_emission_calculation",
    "record_rule_evaluation",
    "record_report_generation",
    "record_classification_confidence",
    "record_processing_duration",
    "update_active_batches",
    "update_total_spend",
    "record_processing_error",
    "record_factor_lookup",
    # Core engines (Layer 2)
    "RecordIngestionEngine",
    "TaxonomyClassifier",
    "Scope3Mapper",
    "EmissionCalculator",
    "RuleEngine",
    "SpendAnalyticsEngine",
    "ReportGeneratorEngine",
    # Layer 2 Enumerations
    "TaxonomySystem",
    "Scope3Category",
    "ConfidenceLevel",
    "RecordStatus",
    "ClassificationMethod",
    "EmissionFactorSource",
    "ReportFormat",
    "RuleConditionType",
    # Layer 2 Core models
    "SpendRecord",
    "TaxonomyCode",
    "ClassificationResult",
    "Scope3Assignment",
    "EmissionFactor",
    "EmissionResult",
    "CategoryRule",
    # Layer 2 Request models
    "IngestRecordsRequest",
    "ClassifyRequest",
    "CalculateEmissionsRequest",
    # Service setup facade
    "SpendCategorizerService",
    "configure_spend_categorizer",
    "get_spend_categorizer",
    "get_router",
    # Response models
    "SpendRecordResponse",
    "ClassificationResponse",
    "Scope3AssignmentResponse",
    "EmissionCalculationResponse",
    "CategoryRuleResponse",
    "AnalyticsResponse",
    "ReportResponse",
    "SpendCategorizerStatisticsResponse",
]
