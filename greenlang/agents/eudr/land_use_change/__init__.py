# -*- coding: utf-8 -*-
"""
Land Use Change Detector Agent - AGENT-EUDR-005

Multi-method land use change detection engine for EU Deforestation
Regulation (EUDR) Articles 2, 9, 10, and 12 compliance. Classifies
land use categories from multi-spectral satellite imagery using spectral,
vegetation index, phenological, texture, and ensemble methods; detects
land use transitions between time periods including deforestation,
degradation, cropland expansion, and urban expansion; analyses temporal
trajectories of vegetation indices to identify abrupt changes, gradual
trends, and recovery patterns; verifies land use state at the EUDR
cutoff date (December 31, 2020) with conservative bias for regulatory
compliance; detects commodity-driven cropland conversions for all seven
EUDR-regulated commodities; assesses conversion risk using an 8-factor
weighted scoring model; analyses urban encroachment and infrastructure
proximity impacts on forested land; and generates provenance-tracked
compliance reports for regulatory submission.

This package contains:
    Foundational modules:
        - models: Pydantic v2 data models for land use classification,
          transition detection, temporal trajectory analysis, cutoff
          verification, cropland conversion, conversion risk, urban
          encroachment, compliance reporting, and batch analysis
        - config: LandUseChangeConfig with GL_EUDR_LUC_ env var support
        - provenance: SHA-256 chain-hashed audit trail tracking
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_luc_ prefix)

    Engine modules:
        - land_use_classifier: Multi-method land use classification
        - transition_detector: Bi-temporal transition detection
        - temporal_trajectory_analyzer: Time series trajectory analysis
        - cutoff_date_verifier: EUDR Article 2 cutoff verification
        - cropland_expansion_detector: Commodity-driven conversion detection
        - conversion_risk_assessor: Multi-factor risk scoring
        - urban_encroachment_analyzer: Infrastructure proximity analysis
        - compliance_reporter: Regulatory report generation

PRD: PRD-AGENT-EUDR-005
Agent ID: GL-EUDR-LUC-005
Regulation: EU 2023/1115 (EUDR) Article 2(1), Article 9, Article 10, Article 12
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.land_use_change import (
    ...     VerificationRequest,
    ...     CutoffVerification,
    ...     ComplianceVerdict,
    ...     LandUseCategory,
    ...     TransitionType,
    ...     EUDRCommodity,
    ... )
    >>> from datetime import date
    >>> request = VerificationRequest(
    ...     parcel_id="parcel-001",
    ...     polygon_wkt="POLYGON((-62.2 -3.4, -62.1 -3.4, -62.1 -3.5, -62.2 -3.5, -62.2 -3.4))",
    ...     commodity=EUDRCommodity.SOYA,
    ...     include_evidence=True,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# ---- Foundational: config ----
from greenlang.agents.eudr.land_use_change.config import (
    LandUseChangeConfig,
    get_config,
    set_config,
    reset_config,
)

# ---- Foundational: models ----
from greenlang.agents.eudr.land_use_change.models import (
    # Constants
    VERSION,
    EUDR_CUTOFF_DATE,
    MAX_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_TRANSITION_AREA_HA,
    MAX_TIME_STEPS,
    # Re-exported from greenlang.eudr_traceability.models
    EUDRCommodity,
    # Enumerations
    LandUseCategory,
    TransitionType,
    TrajectoryType,
    ComplianceVerdict,
    ClassificationMethod,
    ConversionType,
    RiskTier,
    InfrastructureType,
    ReportType,
    ReportFormat,
    DataQualityLevel,
    BatchJobStatus,
    # Core models
    LandUseClassification,
    LandUseTransition,
    TransitionMatrix,
    TemporalTrajectory,
    CutoffVerification,
    CroplandConversion,
    ConversionRisk,
    UrbanEncroachment,
    ComplianceReport,
    BatchJob,
    # Request models
    ClassificationRequest,
    TransitionRequest,
    TrajectoryRequest,
    VerificationRequest,
    RiskAssessmentRequest,
    ReportRequest,
    # Response models
    ClassificationResponse,
    TransitionResponse,
    VerificationResponse,
    BatchResponse,
)

# ---- Foundational: provenance ----
from greenlang.agents.eudr.land_use_change.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---- Foundational: metrics ----
from greenlang.agents.eudr.land_use_change.metrics import (
    PROMETHEUS_AVAILABLE,
    record_classification,
    record_transition,
    record_trajectory,
    record_verification,
    record_conversion,
    record_risk_assessment,
    record_urban_analysis,
    record_report,
    record_error,
    record_batch_job,
    observe_analysis_duration,
    set_active_analyses,
    set_avg_classification_confidence,
    set_avg_transition_magnitude,
    set_data_quality_score,
    set_total_analyzed_area_ha,
    set_conversion_risk_score,
    record_non_compliant_parcel,
)

# ---- Engine modules (optional imports for forward compatibility) ----
try:
    from greenlang.agents.eudr.land_use_change.land_use_classifier import (
        LandUseClassifier,
    )
except ImportError:
    LandUseClassifier = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.transition_detector import (
        TransitionDetector,
    )
except ImportError:
    TransitionDetector = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.temporal_trajectory_analyzer import (
        TemporalTrajectoryAnalyzer,
    )
except ImportError:
    TemporalTrajectoryAnalyzer = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.cutoff_date_verifier import (
        CutoffDateVerifier,
    )
except ImportError:
    CutoffDateVerifier = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.cropland_expansion_detector import (
        CroplandExpansionDetector,
    )
except ImportError:
    CroplandExpansionDetector = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.conversion_risk_assessor import (
        ConversionRiskAssessor,
    )
except ImportError:
    ConversionRiskAssessor = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.urban_encroachment_analyzer import (
        UrbanEncroachmentAnalyzer,
    )
except ImportError:
    UrbanEncroachmentAnalyzer = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.land_use_change.compliance_reporter import (
        ComplianceReporter,
    )
except ImportError:
    ComplianceReporter = None  # type: ignore[assignment, misc]

# ---- Version ----
__version__ = "1.0.0"

__all__ = [
    # -- Version --
    "__version__",
    # -- Config --
    "LandUseChangeConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "DEFAULT_NUM_CLASSES",
    "MIN_CONFIDENCE_THRESHOLD",
    "MIN_TRANSITION_AREA_HA",
    "MAX_TIME_STEPS",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "LandUseCategory",
    "TransitionType",
    "TrajectoryType",
    "ComplianceVerdict",
    "ClassificationMethod",
    "ConversionType",
    "RiskTier",
    "InfrastructureType",
    "ReportType",
    "ReportFormat",
    "DataQualityLevel",
    "BatchJobStatus",
    # -- Core Models --
    "LandUseClassification",
    "LandUseTransition",
    "TransitionMatrix",
    "TemporalTrajectory",
    "CutoffVerification",
    "CroplandConversion",
    "ConversionRisk",
    "UrbanEncroachment",
    "ComplianceReport",
    "BatchJob",
    # -- Request Models --
    "ClassificationRequest",
    "TransitionRequest",
    "TrajectoryRequest",
    "VerificationRequest",
    "RiskAssessmentRequest",
    "ReportRequest",
    # -- Response Models --
    "ClassificationResponse",
    "TransitionResponse",
    "VerificationResponse",
    "BatchResponse",
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
    "record_classification",
    "record_transition",
    "record_trajectory",
    "record_verification",
    "record_conversion",
    "record_risk_assessment",
    "record_urban_analysis",
    "record_report",
    "record_error",
    "record_batch_job",
    "observe_analysis_duration",
    "set_active_analyses",
    "set_avg_classification_confidence",
    "set_avg_transition_magnitude",
    "set_data_quality_score",
    "set_total_analyzed_area_ha",
    "set_conversion_risk_score",
    "record_non_compliant_parcel",
    # -- Engine Classes (optional) --
    "LandUseClassifier",
    "TransitionDetector",
    "TemporalTrajectoryAnalyzer",
    "CutoffDateVerifier",
    "CroplandExpansionDetector",
    "ConversionRiskAssessor",
    "UrbanEncroachmentAnalyzer",
    "ComplianceReporter",
]
