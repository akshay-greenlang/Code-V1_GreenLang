"""
GL-EUDR-001: Supply Chain Mapper Agent

Maps multi-tier supply chains from production plots to EU market entry
for EUDR compliance. This is the foundational agent in the EUDR
Supply Chain Traceability pipeline.

Features:
- Multi-tier supply chain mapping (5-10+ tiers)
- Hybrid entity resolution (deterministic rules + ML)
- Coverage calculation with risk-weighted gates
- PostgreSQL primary storage with Neo4j graph cache
- Temporal tracking with as-of queries and snapshots
- LLM integration for entity extraction and NL queries

Regulatory Reference:
    EU Regulation 2023/1115 (EUDR)
    Enforcement Date: December 30, 2024 (Large Operators)
    SME Enforcement Date: June 30, 2025

Example:
    >>> from backend.agents.gl_eudr_001_supply_chain_mapper import (
    ...     SupplyChainMapperAgent,
    ...     SupplyChainMapperInput,
    ...     CommodityType,
    ...     OperationType,
    ... )
    >>> import uuid
    >>> agent = SupplyChainMapperAgent()
    >>> result = agent.run(SupplyChainMapperInput(
    ...     importer_id=uuid.uuid4(),
    ...     commodity=CommodityType.COFFEE,
    ...     operation=OperationType.MAP_SUPPLY_CHAIN
    ... ))
    >>> print(f"Nodes: {result.node_count}, Coverage: {result.coverage_report}")
"""

from .agent import (
    # Main Agent
    SupplyChainMapperAgent,

    # Input/Output Models
    SupplyChainMapperInput,
    SupplyChainMapperOutput,

    # Core Enums
    CommodityType,
    NodeType,
    EdgeType,
    DataSource,
    VerificationStatus,
    DisclosureStatus,
    OperatorSize,
    GapType,
    GapSeverity,
    ResolutionStatus,
    ResolutionDecision,
    SnapshotTrigger,
    RiskLevel,
    OperationType,

    # Address & Geolocation Models
    Address,
    GeoCoordinate,
    PlotGeometry,

    # Supply Chain Node Models
    Certification,
    SupplyChainNode,

    # Supply Chain Edge Models
    EdgeContext,
    InferenceEvidence,
    SupplyChainEdge,

    # Origin Plot Models
    OriginPlot,
    PlotProducer,

    # Coverage Models
    SupplyChainGap,
    CoverageThresholds,
    CoverageReport,
    CoverageGateResult,
    COVERAGE_GATES,

    # Snapshot Models
    SupplyChainSnapshot,
    SnapshotDiff,

    # Entity Resolution Models
    MatchFeature,
    EntityResolutionCandidate,
    MatchResult,

    # Graph Model
    SupplyChainGraph,

    # Pack Spec
    PACK_SPEC,
)

# Database Integration
from .database import (
    DatabaseConfig,
    get_db_session,
    get_sync_db_session,
    get_db_context,
    get_async_engine,
    get_sync_engine,
    check_database_connection,
    init_database,
    cleanup_database,
    BaseRepository,
    NodeRepository,
    EdgeRepository,
    SnapshotRepository,
    TenantContext,
)

# Authentication & Security
from .auth import (
    UserRole,
    Permission,
    User,
    TokenData,
    get_current_user,
    require_permissions,
    require_role,
    ResourceOwnershipVerifier,
    PIIMasker,
    MassAssignmentProtection,
    RateLimiter,
    create_access_token,
    verify_token,
)

# Audit Logging
from .audit import (
    AuditAction,
    AuditSeverity,
    AuditLogEntry,
    AuditContext,
    AuditLogger,
    AuditMiddleware,
    global_audit_logger,
    get_audit_logger,
    audited,
)

# LLM Integration
from .llm_service import (
    LLMProvider,
    LLMIntegrationService,
    ExtractedEntity,
    FuzzyMatchSuggestion,
    ParsedQuery,
    GapMateriality,
)

__all__ = [
    # Main Agent
    "SupplyChainMapperAgent",

    # Input/Output Models
    "SupplyChainMapperInput",
    "SupplyChainMapperOutput",

    # Core Enums
    "CommodityType",
    "NodeType",
    "EdgeType",
    "DataSource",
    "VerificationStatus",
    "DisclosureStatus",
    "OperatorSize",
    "GapType",
    "GapSeverity",
    "ResolutionStatus",
    "ResolutionDecision",
    "SnapshotTrigger",
    "RiskLevel",
    "OperationType",

    # Address & Geolocation Models
    "Address",
    "GeoCoordinate",
    "PlotGeometry",

    # Supply Chain Node Models
    "Certification",
    "SupplyChainNode",

    # Supply Chain Edge Models
    "EdgeContext",
    "InferenceEvidence",
    "SupplyChainEdge",

    # Origin Plot Models
    "OriginPlot",
    "PlotProducer",

    # Coverage Models
    "SupplyChainGap",
    "CoverageThresholds",
    "CoverageReport",
    "CoverageGateResult",
    "COVERAGE_GATES",

    # Snapshot Models
    "SupplyChainSnapshot",
    "SnapshotDiff",

    # Entity Resolution Models
    "MatchFeature",
    "EntityResolutionCandidate",
    "MatchResult",

    # Graph Model
    "SupplyChainGraph",

    # Pack Spec
    "PACK_SPEC",

    # Database Integration
    "DatabaseConfig",
    "get_db_session",
    "get_sync_db_session",
    "get_db_context",
    "get_async_engine",
    "get_sync_engine",
    "check_database_connection",
    "init_database",
    "cleanup_database",
    "BaseRepository",
    "NodeRepository",
    "EdgeRepository",
    "SnapshotRepository",
    "TenantContext",

    # Authentication & Security
    "UserRole",
    "Permission",
    "User",
    "TokenData",
    "get_current_user",
    "require_permissions",
    "require_role",
    "ResourceOwnershipVerifier",
    "PIIMasker",
    "MassAssignmentProtection",
    "RateLimiter",
    "create_access_token",
    "verify_token",

    # Audit Logging
    "AuditAction",
    "AuditSeverity",
    "AuditLogEntry",
    "AuditContext",
    "AuditLogger",
    "AuditMiddleware",
    "global_audit_logger",
    "get_audit_logger",
    "audited",

    # LLM Integration
    "LLMProvider",
    "LLMIntegrationService",
    "ExtractedEntity",
    "FuzzyMatchSuggestion",
    "ParsedQuery",
    "GapMateriality",
]

__version__ = "1.0.0"
__agent_id__ = "eudr/supply_chain_mapper_v1"
