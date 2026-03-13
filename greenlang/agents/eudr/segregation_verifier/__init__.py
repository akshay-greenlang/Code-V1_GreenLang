# -*- coding: utf-8 -*-
"""
Segregation Verifier Agent - AGENT-EUDR-010

Production-grade physical segregation verification of EUDR-compliant
vs non-compliant material across the supply chain: storage segregation,
transport segregation, processing line verification, cross-contamination
detection, labeling compliance, and facility assessment for the EU
Deforestation Regulation (EUDR).

This package provides a comprehensive segregation verification engine
for EUDR supply chain traceability supporting:

    Segregation Domains:
        - Storage: Zone-based segregation with barrier verification,
          minimum separation distances, and occupancy tracking.
        - Transport: Vehicle cleaning verification, cargo history
          review, seal integrity, and dedicated vehicle tracking.
        - Processing: Line changeover protocols, flush volume
          verification, temporal separation, and first-run flagging.
        - Contamination: Multi-pathway detection (10 pathways),
          severity classification, impact tracing, and auto-downgrade.
        - Labeling: 8 label types, placement verification, expiry
          tracking, and color-coded zone compliance.
        - Facility: 6-level capability assessment with weighted
          scoring across 5 dimensions (layout, protocols, history,
          labeling, documentation).

    Foundational modules:
        - config: SegregationVerifierConfig with GL_EUDR_SGV_ env var
          support covering storage, transport, processing, contamination,
          labeling, assessment, and SCP verification settings
        - models: Pydantic v2 data models with 15 enumerations, 12 core
          models, 15 request models, and 11 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 14 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_sgv_)

    Engine modules:
        - segregation_point_validator: Validates SCP registration and
          compliance scoring
        - storage_segregation_auditor: Audits storage zone barriers,
          separation distances, and adjacent zone risks
        - transport_segregation_tracker: Verifies vehicle cleaning,
          cargo history, seals, and dedication status
        - processing_line_verifier: Validates changeover protocols,
          flush volumes, and temporal separation
        - cross_contamination_detector: Detects contamination via 10
          pathway types with configurable proximity thresholds
        - labeling_verification_engine: Verifies label presence,
          readability, placement, and content compliance
        - facility_assessment_engine: Conducts weighted 5-dimension
          facility capability assessments
        - compliance_reporter: Generates Article 9/14 compliance
          reports in JSON, PDF, CSV, and EUDR XML formats

PRD: PRD-AGENT-EUDR-010
Agent ID: GL-EUDR-SGV-010
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14, 31
Standard: ISO 22095:2020 Chain of Custody - Physical Segregation
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.segregation_verifier import (
    ...     SegregationControlPoint,
    ...     SCPType,
    ...     SCPStatus,
    ...     SegregationMethod,
    ...     SegregationVerifierConfig,
    ...     get_config,
    ... )
    >>> scp = SegregationControlPoint(
    ...     facility_id="fac-001",
    ...     scp_type=SCPType.STORAGE,
    ...     commodity="cocoa",
    ...     segregation_method=SegregationMethod.PHYSICAL_BARRIER,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-SGV-010"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.segregation_verifier.config import (
        SegregationVerifierConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    SegregationVerifierConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.segregation_verifier.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_SEGREGATION_POINTS,
        MAX_STORAGE_ZONES,
        MAX_TRANSPORT_VEHICLES,
        DEFAULT_TEMPORAL_PROXIMITY_HOURS,
        DEFAULT_SPATIAL_PROXIMITY_METERS,
        DEFAULT_REVERIFICATION_DAYS,
        DEFAULT_MIN_CHANGEOVER_MINUTES,
        EUDR_RETENTION_YEARS,
        MAX_PREVIOUS_CARGOES,
        MAX_CONTAMINATION_DEPTH,
        ASSESSMENT_LAYOUT_WEIGHT,
        ASSESSMENT_PROTOCOL_WEIGHT,
        ASSESSMENT_HISTORY_WEIGHT,
        ASSESSMENT_LABELING_WEIGHT,
        ASSESSMENT_DOCUMENTATION_WEIGHT,
        MAX_BATCH_SIZE,
        PRIMARY_COMMODITIES,
        FACILITY_ASSESSMENT_LEVELS,
        # Enumerations
        SCPType,
        SCPStatus,
        SegregationMethod,
        StorageType,
        TransportType,
        ProcessingLineType,
        ContaminationPathway,
        ContaminationSeverity,
        LabelType,
        LabelStatus,
        FacilityCapabilityLevel,
        ReportFormat,
        RiskClassification,
        ComplianceStatus,
        CleaningMethod,
        # Core Models
        SegregationControlPoint,
        StorageZone,
        StorageEvent,
        TransportVehicle,
        TransportVerification,
        ProcessingLine,
        ChangeoverRecord,
        ContaminationEvent,
        LabelRecord,
        FacilityAssessment,
        ContaminationImpact,
        SegregationReport,
        # Request Models
        RegisterSCPRequest,
        ValidateSCPRequest,
        RegisterStorageZoneRequest,
        RecordStorageEventRequest,
        RegisterVehicleRequest,
        VerifyTransportRequest,
        RegisterProcessingLineRequest,
        RecordChangeoverRequest,
        DetectContaminationRequest,
        RecordContaminationRequest,
        VerifyLabelsRequest,
        RunAssessmentRequest,
        GenerateReportRequest,
        SearchSCPRequest,
        BatchImportSCPRequest,
        # Response Models
        SCPResponse,
        StorageAuditResponse,
        TransportVerificationResponse,
        ProcessingVerificationResponse,
        ContaminationDetectionResponse,
        ContaminationImpactResponse,
        LabelAuditResponse,
        AssessmentResponse,
        ReportResponse,
        BatchJobResponse,
        HealthResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.segregation_verifier.provenance import (
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
    from greenlang.agents.eudr.segregation_verifier.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        sgv_scp_validations_total,
        sgv_scp_failures_total,
        sgv_storage_audits_total,
        sgv_transport_checks_total,
        sgv_processing_checks_total,
        sgv_contamination_events_total,
        sgv_contamination_critical_total,
        sgv_labels_verified_total,
        sgv_label_failures_total,
        sgv_assessments_total,
        sgv_reports_generated_total,
        sgv_batch_jobs_total,
        sgv_api_errors_total,
        sgv_scp_validation_duration_seconds,
        sgv_contamination_detection_duration_seconds,
        sgv_assessment_duration_seconds,
        sgv_active_segregation_points,
        sgv_avg_facility_score,
        # Helper functions
        record_scp_validation,
        record_scp_failure,
        record_storage_audit,
        record_transport_check,
        record_processing_check,
        record_contamination_event,
        record_contamination_critical,
        record_label_verified,
        record_label_failure,
        record_assessment,
        record_report_generated,
        record_batch_job,
        record_api_error,
        observe_scp_validation_duration,
        observe_contamination_detection_duration,
        observe_assessment_duration,
        set_active_segregation_points,
        set_avg_facility_score,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---- Engine 1: Segregation Point Validator ----
try:
    from greenlang.agents.eudr.segregation_verifier.segregation_point_validator import (  # noqa: E501
        SegregationPointValidator,
    )
except ImportError:
    SegregationPointValidator = None  # type: ignore[assignment,misc]

# ---- Engine 2: Storage Segregation Auditor ----
try:
    from greenlang.agents.eudr.segregation_verifier.storage_segregation_auditor import (  # noqa: E501
        StorageSegregationAuditor,
    )
except ImportError:
    StorageSegregationAuditor = None  # type: ignore[assignment,misc]

# ---- Engine 3: Transport Segregation Tracker ----
try:
    from greenlang.agents.eudr.segregation_verifier.transport_segregation_tracker import (  # noqa: E501
        TransportSegregationTracker,
    )
except ImportError:
    TransportSegregationTracker = None  # type: ignore[assignment,misc]

# ---- Engine 4: Processing Line Verifier ----
try:
    from greenlang.agents.eudr.segregation_verifier.processing_line_verifier import (  # noqa: E501
        ProcessingLineVerifier,
    )
except ImportError:
    ProcessingLineVerifier = None  # type: ignore[assignment,misc]

# ---- Engine 5: Cross Contamination Detector ----
try:
    from greenlang.agents.eudr.segregation_verifier.cross_contamination_detector import (  # noqa: E501
        CrossContaminationDetector,
    )
except ImportError:
    CrossContaminationDetector = None  # type: ignore[assignment,misc]

# ---- Engine 6: Labeling Verification Engine ----
try:
    from greenlang.agents.eudr.segregation_verifier.labeling_verification_engine import (  # noqa: E501
        LabelingVerificationEngine,
    )
except ImportError:
    LabelingVerificationEngine = None  # type: ignore[assignment,misc]

# ---- Engine 7: Facility Assessment Engine ----
try:
    from greenlang.agents.eudr.segregation_verifier.facility_assessment_engine import (  # noqa: E501
        FacilityAssessmentEngine,
    )
except ImportError:
    FacilityAssessmentEngine = None  # type: ignore[assignment,misc]

# ---- Engine 8: Compliance Reporter ----
try:
    from greenlang.agents.eudr.segregation_verifier.compliance_reporter import (
        ComplianceReporter,
    )
except ImportError:
    ComplianceReporter = None  # type: ignore[assignment,misc]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "SegregationVerifierConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_SEGREGATION_POINTS",
    "MAX_STORAGE_ZONES",
    "MAX_TRANSPORT_VEHICLES",
    "DEFAULT_TEMPORAL_PROXIMITY_HOURS",
    "DEFAULT_SPATIAL_PROXIMITY_METERS",
    "DEFAULT_REVERIFICATION_DAYS",
    "DEFAULT_MIN_CHANGEOVER_MINUTES",
    "EUDR_RETENTION_YEARS",
    "MAX_PREVIOUS_CARGOES",
    "MAX_CONTAMINATION_DEPTH",
    "ASSESSMENT_LAYOUT_WEIGHT",
    "ASSESSMENT_PROTOCOL_WEIGHT",
    "ASSESSMENT_HISTORY_WEIGHT",
    "ASSESSMENT_LABELING_WEIGHT",
    "ASSESSMENT_DOCUMENTATION_WEIGHT",
    "MAX_BATCH_SIZE",
    "PRIMARY_COMMODITIES",
    "FACILITY_ASSESSMENT_LEVELS",
    # -- Enumerations --
    "SCPType",
    "SCPStatus",
    "SegregationMethod",
    "StorageType",
    "TransportType",
    "ProcessingLineType",
    "ContaminationPathway",
    "ContaminationSeverity",
    "LabelType",
    "LabelStatus",
    "FacilityCapabilityLevel",
    "ReportFormat",
    "RiskClassification",
    "ComplianceStatus",
    "CleaningMethod",
    # -- Core Models --
    "SegregationControlPoint",
    "StorageZone",
    "StorageEvent",
    "TransportVehicle",
    "TransportVerification",
    "ProcessingLine",
    "ChangeoverRecord",
    "ContaminationEvent",
    "LabelRecord",
    "FacilityAssessment",
    "ContaminationImpact",
    "SegregationReport",
    # -- Request Models --
    "RegisterSCPRequest",
    "ValidateSCPRequest",
    "RegisterStorageZoneRequest",
    "RecordStorageEventRequest",
    "RegisterVehicleRequest",
    "VerifyTransportRequest",
    "RegisterProcessingLineRequest",
    "RecordChangeoverRequest",
    "DetectContaminationRequest",
    "RecordContaminationRequest",
    "VerifyLabelsRequest",
    "RunAssessmentRequest",
    "GenerateReportRequest",
    "SearchSCPRequest",
    "BatchImportSCPRequest",
    # -- Response Models --
    "SCPResponse",
    "StorageAuditResponse",
    "TransportVerificationResponse",
    "ProcessingVerificationResponse",
    "ContaminationDetectionResponse",
    "ContaminationImpactResponse",
    "LabelAuditResponse",
    "AssessmentResponse",
    "ReportResponse",
    "BatchJobResponse",
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
    "sgv_scp_validations_total",
    "sgv_scp_failures_total",
    "sgv_storage_audits_total",
    "sgv_transport_checks_total",
    "sgv_processing_checks_total",
    "sgv_contamination_events_total",
    "sgv_contamination_critical_total",
    "sgv_labels_verified_total",
    "sgv_label_failures_total",
    "sgv_assessments_total",
    "sgv_reports_generated_total",
    "sgv_batch_jobs_total",
    "sgv_api_errors_total",
    "sgv_scp_validation_duration_seconds",
    "sgv_contamination_detection_duration_seconds",
    "sgv_assessment_duration_seconds",
    "sgv_active_segregation_points",
    "sgv_avg_facility_score",
    "record_scp_validation",
    "record_scp_failure",
    "record_storage_audit",
    "record_transport_check",
    "record_processing_check",
    "record_contamination_event",
    "record_contamination_critical",
    "record_label_verified",
    "record_label_failure",
    "record_assessment",
    "record_report_generated",
    "record_batch_job",
    "record_api_error",
    "observe_scp_validation_duration",
    "observe_contamination_detection_duration",
    "observe_assessment_duration",
    "set_active_segregation_points",
    "set_avg_facility_score",
    # -- Engine 1: Segregation Point Validator --
    "SegregationPointValidator",
    # -- Engine 2: Storage Segregation Auditor --
    "StorageSegregationAuditor",
    # -- Engine 3: Transport Segregation Tracker --
    "TransportSegregationTracker",
    # -- Engine 4: Processing Line Verifier --
    "ProcessingLineVerifier",
    # -- Engine 5: Cross Contamination Detector --
    "CrossContaminationDetector",
    # -- Engine 6: Labeling Verification Engine --
    "LabelingVerificationEngine",
    # -- Engine 7: Facility Assessment Engine --
    "FacilityAssessmentEngine",
    # -- Engine 8: Compliance Reporter --
    "ComplianceReporter",
]
