# -*- coding: utf-8 -*-
"""
Supply Chain Mapping Master Agent - AGENT-EUDR-001

Graph-native supply chain modeling engine for EU Deforestation Regulation
(EUDR) compliance. Provides multi-tier recursive mapping, plot-to-product
traceability, risk propagation, gap analysis, and regulatory export
capabilities for all seven EUDR-regulated commodities.

This package contains:
    Foundational modules:
        - models: Pydantic v2 data models for graph nodes, edges, gaps, API
        - config: SupplyChainMapperConfig with GL_EUDR_SCM_ env var support
        - provenance: SHA-256 chain-hashed audit trail tracking
        - metrics: 15 Prometheus self-monitoring metrics (gl_eudr_scm_ prefix)
    Engine modules:
        - graph_engine: Core DAG graph engine for supply chain modeling
        - geolocation_linker: Plot-level geolocation integration (Feature 3)
        - risk_propagation: Deterministic risk propagation engine
        - multi_tier_mapper: Recursive supply chain discovery
        - supplier_onboarding: Supplier onboarding workflow (Feature 8)

PRD: PRD-AGENT-EUDR-001
Agent ID: GL-EUDR-SCM-001
Regulation: EU 2023/1115 (EUDR)

Example:
    >>> from greenlang.agents.eudr.supply_chain_mapper import (
    ...     SupplyChainNode,
    ...     SupplyChainEdge,
    ...     SupplyChainGraph,
    ...     NodeType,
    ...     EUDRCommodity,
    ... )
    >>> from greenlang.agents.eudr.supply_chain_mapper.models import (
    ...     RiskLevel as ModelsRiskLevel,
    ... )
    >>> node = SupplyChainNode(
    ...     node_type=NodeType.PRODUCER,
    ...     operator_id="op-001",
    ...     operator_name="Fazenda Verde",
    ...     country_code="BR",
    ...     commodities=[EUDRCommodity.SOYA],
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# ---- Engine modules (existing) ----
from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
    SupplyChainGraphEngine,
)
from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
    GeolocationLinker,
    PostGISQueryBuilder,
    CoordinateValidation,
    PolygonValidation,
    DistanceMetric,
    LinkageStatus,
    GeolocationGapType,
    GeolocationGapSeverity,
    ProtectedAreaType,
)
from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
    RiskPropagationEngine,
    RiskPropagationConfig,
    NodeRiskInput,
    NodeRiskResult,
    PropagationResult,
    RiskLevel,
)
from greenlang.agents.eudr.supply_chain_mapper.multi_tier_mapper import (
    MultiTierMapper,
    MultiTierMappingInput,
    MultiTierMappingOutput,
    SupplierRecord,
    TierMappingResult,
    TierDepthDistribution,
    OpaqueSegment,
    CommodityArchetype,
    COMMODITY_ARCHETYPES,
    MappingSourceType,
    OpaqueReason,
    DiscoveryStatus,
    ERPConnectorProtocol,
    QuestionnaireProcessorProtocol,
    PDFExtractorProtocol,
    BulkImporterProtocol,
    GraphStorageProtocol,
)

# ---- Gap Analyzer (Feature 6) ----
from greenlang.agents.eudr.supply_chain_mapper.gap_analyzer import (
    GapAnalyzer,
    GapAnalyzerConfig,
    DetectedGap as GapAnalyzerDetectedGap,
    RemediationAction,
    GapTrendSnapshot,
    GapAnalysisResult as GapAnalyzerResult,
)

# ---- Supplier Onboarding (Feature 8) ----
from greenlang.agents.eudr.supply_chain_mapper.supplier_onboarding import (
    SupplierOnboardingEngine,
    OnboardingSession,
    OnboardingStatus,
    OnboardingStepResult,
    PlotData as OnboardingPlotData,
    CertificationData as OnboardingCertificationData,
    SubTierSupplierData,
    ReminderRecord,
    ReminderType,
    ReminderStatus,
    BulkImportResult,
    BulkImportStatus,
    OnboardingMetrics,
    WIZARD_STEPS,
    SUPPORTED_LANGUAGES,
)

# ---- Regulatory Exporter (Feature 9) ----
from greenlang.agents.eudr.supply_chain_mapper.regulatory_exporter import (
    RegulatoryExporter,
    DDSSchemaValidator,
    DDSXMLSerializer,
    PDFReportGenerator,
    OperatorInfo,
    ProductInfo,
    DeclarationInfo,
    RiskAssessmentInfo,
    DDSValidationResult,
    DDSExportResult,
    BatchExportResult,
    PDFReportResult,
    IncrementalExportResult,
    ExportFormat as DDSExportFormat,
    ExportStatus as DDSExportStatus,
    SubmissionStatus as DDSSubmissionStatus,
    DDS_JSON_SCHEMA,
    DDS_SCHEMA_VERSION,
    EUDR_REGULATION_REF,
    ARTICLE_4_2_FIELDS,
    create_exporter,
)

# ---- Visualization Engine (Feature 7) ----
from greenlang.agents.eudr.supply_chain_mapper.visualization_engine import (
    VisualizationEngine,
    VisualizationConfig,
    LayoutAlgorithm,
    ExportFormat,
    ColorScheme,
    NodePosition,
    EdgePath,
    ClusterGroup,
    SankeyNode,
    SankeyLink,
    LayoutResult,
    SankeyResult,
    GraphFilter,
)

# ---- Foundational: config ----
from greenlang.agents.eudr.supply_chain_mapper.config import (
    SupplyChainMapperConfig,
    get_config,
    set_config,
    reset_config,
)

# ---- Foundational: models ----
from greenlang.agents.eudr.supply_chain_mapper.models import (
    # Constants
    VERSION,
    MAX_NODES_PER_GRAPH,
    MAX_EDGES_PER_GRAPH,
    MAX_TIER_DEPTH,
    EUDR_DEFORESTATION_CUTOFF,
    DEFAULT_RISK_WEIGHTS,
    DERIVED_TO_PRIMARY,
    PRIMARY_COMMODITIES,
    GAP_SEVERITY_MAP,
    GAP_ARTICLE_MAP,
    # Enumerations
    NodeType,
    EUDRCommodity,
    CustodyModel,
    ComplianceStatus,
    GapType,
    GapSeverity,
    TransportMode,
    # Core models
    SupplyChainNode,
    SupplyChainEdge,
    SupplyChainGraph,
    SupplyChainGap,
    RiskPropagationResult,
    # Request models
    CreateGraphRequest,
    CreateNodeRequest,
    CreateEdgeRequest,
    UpdateNodeRequest,
    # Query parameter models
    GraphQueryParams,
    NodeQueryParams,
    EdgeQueryParams,
    # Response models
    TraceResult,
    TierDistribution,
    RiskSummary,
    GapAnalysisResult,
    DDSExportData,
    GraphLayoutData,
    SankeyData,
)

# ---- Foundational: provenance ----
from greenlang.agents.eudr.supply_chain_mapper.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---- Service Facade: setup ----
from greenlang.agents.eudr.supply_chain_mapper.setup import (
    SupplyChainMapperService,
    HealthStatus,
    lifespan,
    get_service,
    set_service,
    reset_service,
)

# ---- Foundational: metrics ----
from greenlang.agents.eudr.supply_chain_mapper.metrics import (
    PROMETHEUS_AVAILABLE,
    record_graph_created,
    record_node_added,
    record_edge_added,
    record_tier_discovery,
    record_trace_operation,
    record_risk_propagation,
    record_gap_detected,
    record_gap_resolved,
    record_dds_export,
    observe_processing_duration,
    observe_graph_query_duration,
    record_error,
    set_active_graphs,
    set_total_nodes,
    set_compliance_readiness_avg,
)

__all__ = [
    # -- Engine modules (existing) --
    "SupplyChainGraphEngine",
    # Geolocation Linker (Feature 3)
    "GeolocationLinker",
    "PostGISQueryBuilder",
    "CoordinateValidation",
    "PolygonValidation",
    "DistanceMetric",
    "LinkageStatus",
    "GeolocationGapType",
    "GeolocationGapSeverity",
    "ProtectedAreaType",
    # Risk Propagation (Feature 5)
    "RiskPropagationEngine",
    "RiskPropagationConfig",
    "NodeRiskInput",
    "NodeRiskResult",
    "PropagationResult",
    "RiskLevel",
    # -- Version --
    "VERSION",
    # -- Config --
    "SupplyChainMapperConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "MAX_NODES_PER_GRAPH",
    "MAX_EDGES_PER_GRAPH",
    "MAX_TIER_DEPTH",
    "EUDR_DEFORESTATION_CUTOFF",
    "DEFAULT_RISK_WEIGHTS",
    "DERIVED_TO_PRIMARY",
    "PRIMARY_COMMODITIES",
    "GAP_SEVERITY_MAP",
    "GAP_ARTICLE_MAP",
    # -- Enumerations (from models.py) --
    "NodeType",
    "EUDRCommodity",
    "CustodyModel",
    "ComplianceStatus",
    "GapType",
    "GapSeverity",
    "TransportMode",
    # -- Core Models --
    "SupplyChainNode",
    "SupplyChainEdge",
    "SupplyChainGraph",
    "SupplyChainGap",
    "RiskPropagationResult",
    # -- Request Models --
    "CreateGraphRequest",
    "CreateNodeRequest",
    "CreateEdgeRequest",
    "UpdateNodeRequest",
    # -- Query Models --
    "GraphQueryParams",
    "NodeQueryParams",
    "EdgeQueryParams",
    # -- Response Models --
    "TraceResult",
    "TierDistribution",
    "RiskSummary",
    "GapAnalysisResult",
    "DDSExportData",
    "GraphLayoutData",
    "SankeyData",
    # -- Provenance --
    "ProvenanceEntry",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Multi-Tier Mapper (Feature 2) --
    "MultiTierMapper",
    "MultiTierMappingInput",
    "MultiTierMappingOutput",
    "SupplierRecord",
    "TierMappingResult",
    "TierDepthDistribution",
    "OpaqueSegment",
    "CommodityArchetype",
    "COMMODITY_ARCHETYPES",
    "MappingSourceType",
    "OpaqueReason",
    "DiscoveryStatus",
    "ERPConnectorProtocol",
    "QuestionnaireProcessorProtocol",
    "PDFExtractorProtocol",
    "BulkImporterProtocol",
    "GraphStorageProtocol",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_graph_created",
    "record_node_added",
    "record_edge_added",
    "record_tier_discovery",
    "record_trace_operation",
    "record_risk_propagation",
    "record_gap_detected",
    "record_gap_resolved",
    "record_dds_export",
    "observe_processing_duration",
    "observe_graph_query_duration",
    "record_error",
    "set_active_graphs",
    "set_total_nodes",
    "set_compliance_readiness_avg",
    # -- Gap Analyzer (Feature 6) --
    "GapAnalyzer",
    "GapAnalyzerConfig",
    "GapAnalyzerDetectedGap",
    "RemediationAction",
    "GapTrendSnapshot",
    "GapAnalyzerResult",
    # -- Visualization Engine (Feature 7) --
    "VisualizationEngine",
    "VisualizationConfig",
    "LayoutAlgorithm",
    "ExportFormat",
    "ColorScheme",
    "NodePosition",
    "EdgePath",
    "ClusterGroup",
    "SankeyNode",
    "SankeyLink",
    "LayoutResult",
    "SankeyResult",
    "GraphFilter",
    # -- Service Facade (setup.py) --
    "SupplyChainMapperService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # -- Supplier Onboarding (Feature 8) --
    "SupplierOnboardingEngine",
    "OnboardingSession",
    "OnboardingStatus",
    "OnboardingStepResult",
    "OnboardingPlotData",
    "OnboardingCertificationData",
    "SubTierSupplierData",
    "ReminderRecord",
    "ReminderType",
    "ReminderStatus",
    "BulkImportResult",
    "BulkImportStatus",
    "OnboardingMetrics",
    "WIZARD_STEPS",
    "SUPPORTED_LANGUAGES",
    # -- Regulatory Exporter (Feature 9) --
    "RegulatoryExporter",
    "DDSSchemaValidator",
    "DDSXMLSerializer",
    "PDFReportGenerator",
    "OperatorInfo",
    "ProductInfo",
    "DeclarationInfo",
    "RiskAssessmentInfo",
    "DDSValidationResult",
    "DDSExportResult",
    "BatchExportResult",
    "PDFReportResult",
    "IncrementalExportResult",
    "DDSExportFormat",
    "DDSExportStatus",
    "DDSSubmissionStatus",
    "DDS_JSON_SCHEMA",
    "DDS_SCHEMA_VERSION",
    "EUDR_REGULATION_REF",
    "ARTICLE_4_2_FIELDS",
    "create_exporter",
]
