# -*- coding: utf-8 -*-
"""
GL-MRV-SCOPE1-002: GreenLang Refrigerants & F-Gas Agent Service SDK
====================================================================

This package provides Scope 1 GHG emissions calculation from refrigerant
and fluorinated gas sources for the GreenLang framework. It supports:

- Comprehensive refrigerant database covering HFCs, HFC blends, HFOs,
  PFCs, SF6, NF3, HCFCs, CFCs, and natural refrigerants with 50+
  refrigerant types and GWP values from AR4, AR5, AR6, and AR6 20-year
- Equipment-based and mass-balance calculation methods per GHG Protocol
  Chapter 8 and EPA 40 CFR Part 98 Subpart DD/OO/L
- Blend decomposition engine that resolves blended refrigerants (R-404A,
  R-410A, R-407C, etc.) into constituent gases with weight-fraction GWP
- Equipment registry for 15 equipment types (commercial refrigeration,
  HVAC, chillers, heat pumps, transport, switchgear, semiconductor, etc.)
- Leak rate estimation engine with base rates, age factors, climate
  factors, and LDAR adjustment per IPCC and EPA defaults
- Service event tracking (installation, recharge, repair, recovery,
  leak check, decommissioning, conversion) for full lifecycle accounting
- Monte Carlo uncertainty quantification (5000+ iterations) with
  analytical error propagation and data quality scoring (1-5 scale)
- Multi-framework regulatory compliance mapping (GHG Protocol, ISO 14064,
  CSRD/ESRS E1, EPA 40 CFR Part 98 Subpart DD/OO/L, EU F-Gas 2024/573,
  Kigali Amendment, UK F-Gas)
- Phase-down schedule tracking per EU F-Gas, Kigali A5, Kigali non-A5
- GWP application (AR4, AR5, AR6, 20yr/100yr timeframes)
- Complete SHA-256 provenance chain tracking for audit trails
- 12 Prometheus metrics with gl_rf_ prefix for observability
- FastAPI REST API with endpoints at /api/v1/refrigerants-fgas
- Thread-safe configuration with GL_REFRIGERANTS_FGAS_ env prefix
- 7-engine architecture: RefrigerantDatabaseEngine,
  EmissionCalculatorEngine, EquipmentRegistryEngine,
  LeakRateEstimatorEngine, UncertaintyQuantifierEngine,
  ComplianceTrackerEngine, RefrigerantPipelineEngine

Key Components:
    - config: RefrigerantsFGasConfig with GL_REFRIGERANTS_FGAS_ env prefix
    - models: Pydantic v2 models (15 enums, 14+ data models)
    - refrigerant_database: Refrigerant registry and GWP lookup engine
    - emission_calculator: Core refrigerant emissions calculation engine
    - equipment_registry: Equipment type profiling and charge tracking engine
    - leak_rate_estimator: Leak rate estimation by equipment and lifecycle stage
    - uncertainty_quantifier: Monte Carlo and analytical uncertainty engine
    - compliance_tracker: Regulatory compliance and phase-down tracking engine
    - refrigerant_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_rf_ prefix
    - api: FastAPI HTTP service with endpoints
    - setup: RefrigerantsFGasService facade

Example:
    >>> from greenlang.refrigerants_fgas import RefrigerantsFGasService
    >>> service = RefrigerantsFGasService()
    >>> result = service.calculate(
    ...     refrigerant_type="R_410A",
    ...     charge_kg=5.0,
    ...     method="equipment_based",
    ... )
    >>> print(result["total_emissions_tco2e"])

Agent ID: GL-MRV-SCOPE1-002
Agent Name: Refrigerants & F-Gas Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE1-002"
__agent_name__ = "Refrigerants & F-Gas Agent"

# SDK availability flag
REFRIGERANTS_FGAS_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.refrigerants_fgas.config import (
    RefrigerantsFGasConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.refrigerants_fgas.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    get_provenance_tracker,
    reset_provenance_tracker,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
try:
    from greenlang.refrigerants_fgas.metrics import (
        PROMETHEUS_AVAILABLE,
        rf_calculations_total,
        rf_emissions_kg_co2e_total,
        rf_refrigerant_lookups_total,
        rf_leak_rate_selections_total,
        rf_equipment_events_total,
        rf_uncertainty_runs_total,
        rf_compliance_checks_total,
        rf_batch_jobs_total,
        rf_calculation_duration_seconds,
        rf_batch_size,
        rf_active_calculations,
        rf_refrigerants_loaded,
        record_calculation,
        record_emissions,
        record_refrigerant_lookup,
        record_leak_rate_selection,
        record_equipment_event,
        record_uncertainty,
        record_compliance_check,
        record_batch,
        observe_calculation_duration,
        observe_batch_size,
        set_active_calculations,
        set_refrigerants_loaded,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    rf_calculations_total = None  # type: ignore[assignment]
    rf_emissions_kg_co2e_total = None  # type: ignore[assignment]
    rf_refrigerant_lookups_total = None  # type: ignore[assignment]
    rf_leak_rate_selections_total = None  # type: ignore[assignment]
    rf_equipment_events_total = None  # type: ignore[assignment]
    rf_uncertainty_runs_total = None  # type: ignore[assignment]
    rf_compliance_checks_total = None  # type: ignore[assignment]
    rf_batch_jobs_total = None  # type: ignore[assignment]
    rf_calculation_duration_seconds = None  # type: ignore[assignment]
    rf_batch_size = None  # type: ignore[assignment]
    rf_active_calculations = None  # type: ignore[assignment]
    rf_refrigerants_loaded = None  # type: ignore[assignment]
    record_calculation = None  # type: ignore[assignment]
    record_emissions = None  # type: ignore[assignment]
    record_refrigerant_lookup = None  # type: ignore[assignment]
    record_leak_rate_selection = None  # type: ignore[assignment]
    record_equipment_event = None  # type: ignore[assignment]
    record_uncertainty = None  # type: ignore[assignment]
    record_compliance_check = None  # type: ignore[assignment]
    record_batch = None  # type: ignore[assignment]
    observe_calculation_duration = None  # type: ignore[assignment]
    observe_batch_size = None  # type: ignore[assignment]
    set_active_calculations = None  # type: ignore[assignment]
    set_refrigerants_loaded = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Models (enums, data models) - optional import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.refrigerants_fgas.models import (
        # Enumerations
        RefrigerantCategory,
        RefrigerantType,
        GWPSource,
        GWPTimeframe,
        CalculationMethod,
        EquipmentType,
        EquipmentStatus,
        ServiceEventType,
        LifecycleStage,
        CalculationStatus,
        ReportingPeriod,
        RegulatoryFramework,
        ComplianceStatus,
        PhaseDownSchedule,
        UnitType,
        # Core data models
        GWPValue,
        BlendComponent,
        RefrigerantProperties,
        EquipmentProfile,
        ServiceEvent,
        LeakRateProfile,
        CalculationInput,
        MassBalanceData,
        GasEmission,
        CalculationResult,
        BatchCalculationRequest,
        BatchCalculationResponse,
        UncertaintyResult,
        ComplianceRecord,
    )
except ImportError:
    RefrigerantCategory = None  # type: ignore[assignment, misc]
    RefrigerantType = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    GWPTimeframe = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    EquipmentType = None  # type: ignore[assignment, misc]
    EquipmentStatus = None  # type: ignore[assignment, misc]
    ServiceEventType = None  # type: ignore[assignment, misc]
    LifecycleStage = None  # type: ignore[assignment, misc]
    CalculationStatus = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    RegulatoryFramework = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    PhaseDownSchedule = None  # type: ignore[assignment, misc]
    UnitType = None  # type: ignore[assignment, misc]
    GWPValue = None  # type: ignore[assignment, misc]
    BlendComponent = None  # type: ignore[assignment, misc]
    RefrigerantProperties = None  # type: ignore[assignment, misc]
    EquipmentProfile = None  # type: ignore[assignment, misc]
    ServiceEvent = None  # type: ignore[assignment, misc]
    LeakRateProfile = None  # type: ignore[assignment, misc]
    CalculationInput = None  # type: ignore[assignment, misc]
    MassBalanceData = None  # type: ignore[assignment, misc]
    GasEmission = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    BatchCalculationRequest = None  # type: ignore[assignment, misc]
    BatchCalculationResponse = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    ComplianceRecord = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.refrigerants_fgas.refrigerant_database import RefrigerantDatabaseEngine
except ImportError:
    RefrigerantDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.emission_calculator import EmissionCalculatorEngine
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.equipment_registry import EquipmentRegistryEngine
except ImportError:
    EquipmentRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.leak_rate_estimator import LeakRateEstimatorEngine
except ImportError:
    LeakRateEstimatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.compliance_tracker import ComplianceTrackerEngine
except ImportError:
    ComplianceTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.refrigerant_pipeline import RefrigerantPipelineEngine
except ImportError:
    RefrigerantPipelineEngine = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and response models
# ---------------------------------------------------------------------------
try:
    from greenlang.refrigerants_fgas.setup import (
        RefrigerantsFGasService,
        configure_refrigerants_fgas,
        get_service,
        get_router,
        CalculationResponse,
        BatchResponse,
        RefrigerantResponse,
        RefrigerantListResponse,
        EquipmentResponse,
        LeakRateResponse,
        BlendResponse,
        UncertaintyResponse,
        ComplianceResponse,
        ServiceEventResponse,
        ValidationResponse,
        PipelineResponse,
        HealthResponse,
        StatsResponse,
    )
except ImportError:
    RefrigerantsFGasService = None  # type: ignore[assignment, misc]
    configure_refrigerants_fgas = None  # type: ignore[assignment, misc]
    get_service = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]
    CalculationResponse = None  # type: ignore[assignment, misc]
    BatchResponse = None  # type: ignore[assignment, misc]
    RefrigerantResponse = None  # type: ignore[assignment, misc]
    RefrigerantListResponse = None  # type: ignore[assignment, misc]
    EquipmentResponse = None  # type: ignore[assignment, misc]
    LeakRateResponse = None  # type: ignore[assignment, misc]
    BlendResponse = None  # type: ignore[assignment, misc]
    UncertaintyResponse = None  # type: ignore[assignment, misc]
    ComplianceResponse = None  # type: ignore[assignment, misc]
    ServiceEventResponse = None  # type: ignore[assignment, misc]
    ValidationResponse = None  # type: ignore[assignment, misc]
    PipelineResponse = None  # type: ignore[assignment, misc]
    HealthResponse = None  # type: ignore[assignment, misc]
    StatsResponse = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from mrv.refrigerants_fgas
# ---------------------------------------------------------------------------
try:
    from greenlang.agents.mrv.refrigerants_fgas import RefrigerantsFGasAgent
except ImportError:
    RefrigerantsFGasAgent = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "REFRIGERANTS_FGAS_SDK_AVAILABLE",
    # Configuration
    "RefrigerantsFGasConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceEntry",
    "get_provenance_tracker",
    "reset_provenance_tracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "rf_calculations_total",
    "rf_emissions_kg_co2e_total",
    "rf_refrigerant_lookups_total",
    "rf_leak_rate_selections_total",
    "rf_equipment_events_total",
    "rf_uncertainty_runs_total",
    "rf_compliance_checks_total",
    "rf_batch_jobs_total",
    "rf_calculation_duration_seconds",
    "rf_batch_size",
    "rf_active_calculations",
    "rf_refrigerants_loaded",
    # Metric helper functions
    "record_calculation",
    "record_emissions",
    "record_refrigerant_lookup",
    "record_leak_rate_selection",
    "record_equipment_event",
    "record_uncertainty",
    "record_compliance_check",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_refrigerants_loaded",
    # Enumerations
    "RefrigerantCategory",
    "RefrigerantType",
    "GWPSource",
    "GWPTimeframe",
    "CalculationMethod",
    "EquipmentType",
    "EquipmentStatus",
    "ServiceEventType",
    "LifecycleStage",
    "CalculationStatus",
    "ReportingPeriod",
    "RegulatoryFramework",
    "ComplianceStatus",
    "PhaseDownSchedule",
    "UnitType",
    # Core data models
    "GWPValue",
    "BlendComponent",
    "RefrigerantProperties",
    "EquipmentProfile",
    "ServiceEvent",
    "LeakRateProfile",
    "CalculationInput",
    "MassBalanceData",
    "GasEmission",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResponse",
    "UncertaintyResult",
    "ComplianceRecord",
    # Core engines (Layer 2)
    "RefrigerantDatabaseEngine",
    "EmissionCalculatorEngine",
    "EquipmentRegistryEngine",
    "LeakRateEstimatorEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceTrackerEngine",
    "RefrigerantPipelineEngine",
    # Service setup facade
    "RefrigerantsFGasService",
    "configure_refrigerants_fgas",
    "get_service",
    "get_router",
    # Response models
    "CalculationResponse",
    "BatchResponse",
    "RefrigerantResponse",
    "RefrigerantListResponse",
    "EquipmentResponse",
    "LeakRateResponse",
    "BlendResponse",
    "UncertaintyResponse",
    "ComplianceResponse",
    "ServiceEventResponse",
    "ValidationResponse",
    "PipelineResponse",
    "HealthResponse",
    "StatsResponse",
    # Layer 1 re-export
    "RefrigerantsFGasAgent",
]
