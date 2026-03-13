# -*- coding: utf-8 -*-
"""
Multi-Tier Supplier Tracker - AGENT-EUDR-008

Production-grade multi-tier supplier hierarchy discovery, profile management,
tier depth tracking, relationship lifecycle management, risk propagation,
compliance monitoring, gap analysis, and audit reporting for the EU
Deforestation Regulation (EUDR).

This package provides an 8-engine supplier tracking pipeline for EUDR
supply chain traceability:

    Engine 1 - SupplierDiscoveryEngine:
        Recursive supplier hierarchy discovery from ERP data, supplier
        declarations, questionnaires, shipping documents, and certification
        databases. Supports up to 15 tiers of depth with configurable
        confidence thresholds and deduplication.

    Engine 2 - SupplierProfileManager:
        Comprehensive supplier profile CRUD with legal entity, location,
        commodity, certification, compliance status, and contact data.
        Profile completeness scoring with configurable category weights.

    Engine 3 - TierDepthTracker:
        Tier depth calculation and visibility scoring per supply chain.
        Coverage assessment, benchmark comparison against industry
        averages, and gap tier detection.

    Engine 4 - RelationshipManager:
        Supplier relationship lifecycle management through prospective,
        onboarding, active, suspended, and terminated states. Strength
        scoring, seasonal pattern detection, and conflict detection.

    Engine 5 - RiskPropagationEngine:
        Six-category risk assessment (deforestation proximity, country
        risk, certification gap, compliance history, data quality,
        concentration risk) with upstream propagation via max, weighted
        average, or volume-weighted methods.

    Engine 6 - ComplianceMonitor:
        Continuous compliance monitoring across DDS validity, certification
        status, geolocation coverage, and deforestation-free verification.
        Alert generation for expiry warnings and status changes.

    Engine 7 - GapAnalyzer:
        Data gap detection across supplier profiles and supply chain
        tiers. Severity classification (critical, major, minor) with
        automated remediation plan generation and progress tracking.

    Engine 8 - AuditReporter:
        EUDR Article 14 audit-ready report generation in JSON, PDF, CSV,
        and EUDR XML formats. Tier depth summaries, risk propagation
        reports, gap analysis reports, and DDS readiness reports.

Foundational modules:
    - models: Pydantic v2 data models with 13 enumerations, 9 core models,
      7 request models, and 8 response models
    - config: MultiTierSupplierConfig with GL_EUDR_MST_ env var support
    - provenance: SHA-256 chain-hashed audit trail tracking with 10 entity
      types and 12 actions
    - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_mst_ prefix)

PRD: PRD-AGENT-EUDR-008
Agent ID: GL-EUDR-MST-008
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 14
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.multi_tier_supplier import (
    ...     SupplierProfile,
    ...     SupplierType,
    ...     CommodityType,
    ...     ComplianceStatus,
    ...     MultiTierSupplierConfig,
    ...     get_config,
    ... )
    >>> profile = SupplierProfile(
    ...     legal_name="Amazonia Cocoa Cooperative",
    ...     supplier_type=SupplierType.COOPERATIVE,
    ...     country_iso="BR",
    ...     commodities=[CommodityType.COCOA],
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-MST-008"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.multi_tier_supplier.config import (
        MultiTierSupplierConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    MultiTierSupplierConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.multi_tier_supplier.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        MAX_TIER_DEPTH,
        DEFAULT_PROFILE_COMPLETENESS_WEIGHTS,
        DEFAULT_RISK_CATEGORY_WEIGHTS,
        COMPLIANCE_STATUS_THRESHOLDS,
        EUDR_RETENTION_YEARS,
        TYPICAL_CHAIN_DEPTHS,
        # Re-exported commodity enum
        EUDRCommodity,
        # Enumerations
        SupplierTier,
        RelationshipStatus,
        RelationshipConfidence,
        RiskCategory,
        ComplianceStatus,
        GapSeverity,
        CertificationType,
        SupplierType,
        CommodityType,
        ReportFormat,
        BatchStatus,
        RiskPropagationMethod,
        DiscoverySource,
        # Core models
        CertificationRecord,
        SupplierProfile,
        SupplierRelationship,
        TierDepthResult,
        RiskScore,
        RiskPropagationResult,
        ComplianceCheckResult,
        DataGap,
        RemediationPlan,
        # Request models
        DiscoverSuppliersRequest,
        CreateSupplierRequest,
        AssessTierDepthRequest,
        AssessRiskRequest,
        CheckComplianceRequest,
        AnalyzeGapsRequest,
        GenerateReportRequest,
        # Response models
        DiscoverSuppliersResponse,
        CreateSupplierResponse,
        AssessTierDepthResponse,
        AssessRiskResponse,
        CheckComplianceResponse,
        AnalyzeGapsResponse,
        GenerateReportResponse,
        BatchResult,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.multi_tier_supplier.provenance import (
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
    from greenlang.agents.eudr.multi_tier_supplier.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        mst_suppliers_discovered_total,
        mst_suppliers_onboarded_total,
        mst_relationships_created_total,
        mst_tier_depth_assessments_total,
        mst_risk_assessments_total,
        mst_risk_alerts_total,
        mst_compliance_checks_total,
        mst_compliance_alerts_total,
        mst_gaps_detected_total,
        mst_gaps_remediated_total,
        mst_reports_generated_total,
        mst_batch_jobs_total,
        mst_discovery_duration_seconds,
        mst_risk_propagation_duration_seconds,
        mst_compliance_check_duration_seconds,
        mst_active_suppliers,
        mst_avg_tier_depth,
        mst_api_errors_total,
        # Helper functions
        record_supplier_discovered,
        record_supplier_onboarded,
        record_relationship_created,
        record_tier_depth_assessment,
        record_risk_assessment,
        record_risk_alert,
        record_compliance_check,
        record_compliance_alert,
        record_gap_detected,
        record_gap_remediated,
        record_report_generated,
        record_batch_job,
        observe_discovery_duration,
        observe_risk_propagation_duration,
        observe_compliance_check_duration,
        set_active_suppliers,
        set_avg_tier_depth,
        record_api_error,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "MultiTierSupplierConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "MAX_TIER_DEPTH",
    "DEFAULT_PROFILE_COMPLETENESS_WEIGHTS",
    "DEFAULT_RISK_CATEGORY_WEIGHTS",
    "COMPLIANCE_STATUS_THRESHOLDS",
    "EUDR_RETENTION_YEARS",
    "TYPICAL_CHAIN_DEPTHS",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "SupplierTier",
    "RelationshipStatus",
    "RelationshipConfidence",
    "RiskCategory",
    "ComplianceStatus",
    "GapSeverity",
    "CertificationType",
    "SupplierType",
    "CommodityType",
    "ReportFormat",
    "BatchStatus",
    "RiskPropagationMethod",
    "DiscoverySource",
    # -- Core Models --
    "CertificationRecord",
    "SupplierProfile",
    "SupplierRelationship",
    "TierDepthResult",
    "RiskScore",
    "RiskPropagationResult",
    "ComplianceCheckResult",
    "DataGap",
    "RemediationPlan",
    # -- Request Models --
    "DiscoverSuppliersRequest",
    "CreateSupplierRequest",
    "AssessTierDepthRequest",
    "AssessRiskRequest",
    "CheckComplianceRequest",
    "AnalyzeGapsRequest",
    "GenerateReportRequest",
    # -- Response Models --
    "DiscoverSuppliersResponse",
    "CreateSupplierResponse",
    "AssessTierDepthResponse",
    "AssessRiskResponse",
    "CheckComplianceResponse",
    "AnalyzeGapsResponse",
    "GenerateReportResponse",
    "BatchResult",
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
    "mst_suppliers_discovered_total",
    "mst_suppliers_onboarded_total",
    "mst_relationships_created_total",
    "mst_tier_depth_assessments_total",
    "mst_risk_assessments_total",
    "mst_risk_alerts_total",
    "mst_compliance_checks_total",
    "mst_compliance_alerts_total",
    "mst_gaps_detected_total",
    "mst_gaps_remediated_total",
    "mst_reports_generated_total",
    "mst_batch_jobs_total",
    "mst_discovery_duration_seconds",
    "mst_risk_propagation_duration_seconds",
    "mst_compliance_check_duration_seconds",
    "mst_active_suppliers",
    "mst_avg_tier_depth",
    "mst_api_errors_total",
    "record_supplier_discovered",
    "record_supplier_onboarded",
    "record_relationship_created",
    "record_tier_depth_assessment",
    "record_risk_assessment",
    "record_risk_alert",
    "record_compliance_check",
    "record_compliance_alert",
    "record_gap_detected",
    "record_gap_remediated",
    "record_report_generated",
    "record_batch_job",
    "observe_discovery_duration",
    "observe_risk_propagation_duration",
    "observe_compliance_check_duration",
    "set_active_suppliers",
    "set_avg_tier_depth",
    "record_api_error",
]
