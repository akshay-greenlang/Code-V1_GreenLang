# -*- coding: utf-8 -*-
"""
Geolocation Verification Agent - AGENT-EUDR-002

Multi-layer geospatial verification engine for EU Deforestation Regulation
(EUDR) Article 9 compliance. Validates GPS coordinates, polygon boundaries,
protected area overlaps, deforestation cutoff status, temporal boundary
consistency, and generates composite accuracy scores with provenance-tracked
audit trails for all seven EUDR-regulated commodities.

This package contains:
    Foundational modules:
        - models: Pydantic v2 data models for verification requests/results,
          enumerations, accuracy scores, batch progress, and compliance reports
        - config: GeolocationVerificationConfig with GL_EUDR_GEO_ env var support
        - provenance: SHA-256 chain-hashed audit trail tracking
        - metrics: 15 Prometheus self-monitoring metrics (gl_eudr_geo_ prefix)

PRD: PRD-AGENT-EUDR-002
Agent ID: GL-EUDR-GEO-002
Regulation: EU 2023/1115 (EUDR) Article 9
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.geolocation_verification import (
    ...     VerifyPlotRequest,
    ...     PlotVerificationResult,
    ...     VerificationLevel,
    ...     VerificationStatus,
    ...     QualityTier,
    ...     EUDRCommodity,
    ... )
    >>> request = VerifyPlotRequest(
    ...     plot_id="plot-001",
    ...     coordinates=(-3.4653, -62.2159),
    ...     declared_country_code="BR",
    ...     commodity=EUDRCommodity.SOYA,
    ...     verification_level=VerificationLevel.STANDARD,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# ---- Foundational: config ----
from greenlang.agents.eudr.geolocation_verification.config import (
    GeolocationVerificationConfig,
    get_config,
    set_config,
    reset_config,
)

# ---- Foundational: models ----
from greenlang.agents.eudr.geolocation_verification.models import (
    # Constants
    VERSION,
    EUDR_DEFORESTATION_CUTOFF,
    DEFAULT_SCORE_WEIGHTS,
    QUALITY_TIER_THRESHOLDS,
    MAX_BATCH_SIZE,
    # Re-exported from greenlang.agents.data.eudr_traceability.models
    EUDRCommodity,
    # Enumerations
    VerificationLevel,
    VerificationStatus,
    CoordinateIssueType,
    PolygonIssueType,
    OverlapSeverity,
    DeforestationStatus,
    QualityTier,
    ChangeType,
    # Core models
    CoordinateIssue,
    PolygonIssue,
    RepairSuggestion,
    ProtectedAreaOverlap,
    ProtectedAreaProximity,
    TreeCoverLossEvent,
    # Result models
    CoordinateValidationResult,
    PolygonVerificationResult,
    ProtectedAreaCheckResult,
    DeforestationVerificationResult,
    GeolocationAccuracyScore,
    TemporalChangeResult,
    BoundaryChange,
    # Request models
    VerifyCoordinateRequest,
    VerifyPolygonRequest,
    VerifyPlotRequest,
    BatchVerificationRequest,
    ComplianceReportRequest,
    # Response models
    PlotVerificationResult,
    BatchVerificationResult,
    BatchProgress,
    ComplianceReport,
    ComplianceSummary,
)

# ---- Foundational: provenance ----
from greenlang.agents.eudr.geolocation_verification.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---- Foundational: metrics ----
from greenlang.agents.eudr.geolocation_verification.metrics import (
    PROMETHEUS_AVAILABLE,
    record_coordinate_validated,
    record_polygon_verified,
    record_protected_area_check,
    record_deforestation_check,
    record_plot_verified,
    record_batch_job,
    record_batch_plot_processed,
    record_score_calculated,
    record_compliance_report,
    record_issue_detected,
    observe_verification_duration,
    observe_batch_duration,
    record_error,
    set_active_batch_jobs,
    set_avg_accuracy_score,
)


__all__ = [
    # -- Config --
    "GeolocationVerificationConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Version --
    "VERSION",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "DEFAULT_SCORE_WEIGHTS",
    "QUALITY_TIER_THRESHOLDS",
    "MAX_BATCH_SIZE",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "VerificationLevel",
    "VerificationStatus",
    "CoordinateIssueType",
    "PolygonIssueType",
    "OverlapSeverity",
    "DeforestationStatus",
    "QualityTier",
    "ChangeType",
    # -- Core Models --
    "CoordinateIssue",
    "PolygonIssue",
    "RepairSuggestion",
    "ProtectedAreaOverlap",
    "ProtectedAreaProximity",
    "TreeCoverLossEvent",
    # -- Result Models --
    "CoordinateValidationResult",
    "PolygonVerificationResult",
    "ProtectedAreaCheckResult",
    "DeforestationVerificationResult",
    "GeolocationAccuracyScore",
    "TemporalChangeResult",
    "BoundaryChange",
    # -- Request Models --
    "VerifyCoordinateRequest",
    "VerifyPolygonRequest",
    "VerifyPlotRequest",
    "BatchVerificationRequest",
    "ComplianceReportRequest",
    # -- Response Models --
    "PlotVerificationResult",
    "BatchVerificationResult",
    "BatchProgress",
    "ComplianceReport",
    "ComplianceSummary",
    # -- Provenance --
    "ProvenanceEntry",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_coordinate_validated",
    "record_polygon_verified",
    "record_protected_area_check",
    "record_deforestation_check",
    "record_plot_verified",
    "record_batch_job",
    "record_batch_plot_processed",
    "record_score_calculated",
    "record_compliance_report",
    "record_issue_detected",
    "observe_verification_duration",
    "observe_batch_duration",
    "record_error",
    "set_active_batch_jobs",
    "set_avg_accuracy_score",
]
