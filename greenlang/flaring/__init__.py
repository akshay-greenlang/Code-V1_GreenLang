# -*- coding: utf-8 -*-
"""
GL-MRV-SCOPE1-006: GreenLang Flaring Agent Service SDK
=======================================================

This package provides Scope 1 GHG emissions calculation from flaring
operations at industrial facilities for the GreenLang framework. It supports:

- 8 flare types (elevated steam/air/unassisted, enclosed ground, multi-point
  ground, offshore marine, candlestick, low-pressure) with type-specific
  default combustion efficiencies (95-99%)
- 6 flaring event categories (routine, non-routine, emergency, maintenance,
  pilot/purge, well completion) for regulatory classification
- 4 calculation methods: gas composition analysis, default emission factor,
  engineering estimate, and direct measurement (CEMS/ultrasonic)
- 15 gas composition components (CH4, C2H6, C3H8, C4H10, C5H12, C6+,
  CO2, N2, H2S, H2, CO, C2H4, C3H6, H2O) with component HHVs and
  molecular weights for composition-based calculations
- Combustion efficiency modeling with wind speed, tip velocity, LHV,
  steam/air assist ratio adjustments
- Pilot and purge gas emission accounting
- Multi-GWP support (AR4, AR5, AR6, AR6-20yr) for CO2, CH4, N2O
- Black carbon/soot tracking for SLCP reporting
- Uncombusted hydrocarbon (CH4) slip calculations via (1 - CE)
- OGMP 2.0 five-level reporting hierarchy
- Monte Carlo uncertainty quantification (5000 iterations default)
- Multi-framework regulatory compliance checking:
  GHG Protocol, ISO 14064-1, CSRD/ESRS E1, EPA Subpart W Sec. W.23,
  EU ETS MRR, EU Methane Regulation 2024/1787, World Bank ZRF, OGMP 2.0
- Standard conditions: EPA (60F/14.696 psia) and ISO (15C/101.325 kPa)
- Complete SHA-256 provenance chain tracking for audit trails
- 12 Prometheus metrics with gl_fl_ prefix for observability
- Thread-safe configuration with GL_FLARING_ env prefix
- 7-engine architecture: FlareSystemDatabaseEngine,
  EmissionCalculatorEngine, CombustionEfficiencyEngine,
  FlaringEventTrackerEngine, UncertaintyQuantifierEngine,
  ComplianceCheckerEngine, FlaringPipelineEngine

Key Components:
    - config: FlaringConfig with GL_FLARING_ env prefix
    - models: Pydantic v2 models (16 enums, 16+ data models, GWP/HHV/MW
      constant tables)
    - metrics: 12 Prometheus metrics with gl_fl_ prefix
    - provenance: SHA-256 chain-hashed audit trails with 12 entity types
      and 16 actions

Example:
    >>> from greenlang.flaring import FlaringConfig
    >>> from greenlang.flaring import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_combustion_efficiency)
    AR6 0.98

Agent ID: GL-MRV-SCOPE1-006
Agent Name: Flaring Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE1-006"
__agent_name__ = "Flaring Agent"

# SDK availability flag
FLARING_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.flaring.config import (
    FlaringConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.flaring.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    ProvenanceChain,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    VALID_STEP_NAMES,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
try:
    from greenlang.flaring.metrics import (
        PROMETHEUS_AVAILABLE,
        MetricsCollector,
        fl_calculations_total,
        fl_emissions_kg_co2e_total,
        fl_flare_lookups_total,
        fl_factor_selections_total,
        fl_flaring_events_total,
        fl_uncertainty_runs_total,
        fl_compliance_checks_total,
        fl_batch_jobs_total,
        fl_calculation_duration_seconds,
        fl_batch_size,
        fl_active_calculations,
        fl_flare_systems_registered,
        record_calculation,
        record_emissions,
        record_flare_lookup,
        record_factor_selection,
        record_event,
        record_uncertainty,
        record_compliance,
        record_batch,
        observe_duration,
        observe_batch_size,
        set_active_calculations,
        set_flare_systems_registered,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    MetricsCollector = None  # type: ignore[assignment, misc]
    fl_calculations_total = None  # type: ignore[assignment]
    fl_emissions_kg_co2e_total = None  # type: ignore[assignment]
    fl_flare_lookups_total = None  # type: ignore[assignment]
    fl_factor_selections_total = None  # type: ignore[assignment]
    fl_flaring_events_total = None  # type: ignore[assignment]
    fl_uncertainty_runs_total = None  # type: ignore[assignment]
    fl_compliance_checks_total = None  # type: ignore[assignment]
    fl_batch_jobs_total = None  # type: ignore[assignment]
    fl_calculation_duration_seconds = None  # type: ignore[assignment]
    fl_batch_size = None  # type: ignore[assignment]
    fl_active_calculations = None  # type: ignore[assignment]
    fl_flare_systems_registered = None  # type: ignore[assignment]
    record_calculation = None  # type: ignore[assignment]
    record_emissions = None  # type: ignore[assignment]
    record_flare_lookup = None  # type: ignore[assignment]
    record_factor_selection = None  # type: ignore[assignment]
    record_event = None  # type: ignore[assignment]
    record_uncertainty = None  # type: ignore[assignment]
    record_compliance = None  # type: ignore[assignment]
    record_batch = None  # type: ignore[assignment]
    observe_duration = None  # type: ignore[assignment]
    observe_batch_size = None  # type: ignore[assignment]
    set_active_calculations = None  # type: ignore[assignment]
    set_flare_systems_registered = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Models (enums, constants, data models) - optional import with graceful
# fallback so the SDK can still be imported when pydantic is absent
# ---------------------------------------------------------------------------
try:
    from greenlang.flaring.models import (
        # Constants / lookup tables
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        MAX_GASES_PER_RESULT,
        MAX_TRACE_STEPS,
        MAX_COMPOSITION_COMPONENTS,
        CO2_MOLECULAR_WEIGHT,
        CARBON_ATOMIC_WEIGHT,
        GWP_VALUES,
        COMPONENT_HHV_BTU_SCF,
        COMPONENT_MOLECULAR_WEIGHTS,
        COMPONENT_CARBON_COUNT,
        DEFAULT_COMBUSTION_EFFICIENCY,
        # Enumerations (16)
        FlareType,
        FlaringEventCategory,
        CalculationMethod,
        EmissionFactorSource,
        GasComponent,
        EmissionGas,
        GWPSource,
        StandardCondition,
        AssistType,
        FlaringStatus,
        OGMPLevel,
        ComplianceFramework,
        CalculationStatus,
        DataQualityTier,
        SeverityLevel,
        ComplianceStatus,
        # Data models (16+)
        GasComposition,
        FlareSystemConfig,
        FlaringEventRecord,
        PilotPurgeConfig,
        CombustionEfficiencyParams,
        CalculationInput,
        CalculationResult,
        EmissionDetail,
        BatchCalculationRequest,
        BatchCalculationResponse,
        UncertaintyInput,
        UncertaintyResult,
        ComplianceCheckInput,
        ComplianceCheckResult,
        FlareSystemRegistration,
        FlaringStats,
        HealthResponse as ModelsHealthResponse,
    )
except ImportError:
    # Constants
    VERSION = None  # type: ignore[assignment]
    MAX_CALCULATIONS_PER_BATCH = None  # type: ignore[assignment]
    MAX_GASES_PER_RESULT = None  # type: ignore[assignment]
    MAX_TRACE_STEPS = None  # type: ignore[assignment]
    MAX_COMPOSITION_COMPONENTS = None  # type: ignore[assignment]
    CO2_MOLECULAR_WEIGHT = None  # type: ignore[assignment]
    CARBON_ATOMIC_WEIGHT = None  # type: ignore[assignment]
    GWP_VALUES = None  # type: ignore[assignment]
    COMPONENT_HHV_BTU_SCF = None  # type: ignore[assignment]
    COMPONENT_MOLECULAR_WEIGHTS = None  # type: ignore[assignment]
    COMPONENT_CARBON_COUNT = None  # type: ignore[assignment]
    DEFAULT_COMBUSTION_EFFICIENCY = None  # type: ignore[assignment]
    # Enumerations (16)
    FlareType = None  # type: ignore[assignment, misc]
    FlaringEventCategory = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    EmissionFactorSource = None  # type: ignore[assignment, misc]
    GasComponent = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    StandardCondition = None  # type: ignore[assignment, misc]
    AssistType = None  # type: ignore[assignment, misc]
    FlaringStatus = None  # type: ignore[assignment, misc]
    OGMPLevel = None  # type: ignore[assignment, misc]
    ComplianceFramework = None  # type: ignore[assignment, misc]
    CalculationStatus = None  # type: ignore[assignment, misc]
    DataQualityTier = None  # type: ignore[assignment, misc]
    SeverityLevel = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    # Data models (16+)
    GasComposition = None  # type: ignore[assignment, misc]
    FlareSystemConfig = None  # type: ignore[assignment, misc]
    FlaringEventRecord = None  # type: ignore[assignment, misc]
    PilotPurgeConfig = None  # type: ignore[assignment, misc]
    CombustionEfficiencyParams = None  # type: ignore[assignment, misc]
    CalculationInput = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    EmissionDetail = None  # type: ignore[assignment, misc]
    BatchCalculationRequest = None  # type: ignore[assignment, misc]
    BatchCalculationResponse = None  # type: ignore[assignment, misc]
    UncertaintyInput = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    ComplianceCheckInput = None  # type: ignore[assignment, misc]
    ComplianceCheckResult = None  # type: ignore[assignment, misc]
    FlareSystemRegistration = None  # type: ignore[assignment, misc]
    FlaringStats = None  # type: ignore[assignment, misc]
    ModelsHealthResponse = None  # type: ignore[assignment, misc]

# Re-export HealthResponse under its canonical name
HealthResponse = ModelsHealthResponse

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
except ImportError:
    FlareSystemDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.flaring_event_tracker import FlaringEventTrackerEngine
except ImportError:
    FlaringEventTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.flaring_pipeline import FlaringPipelineEngine
except ImportError:
    FlaringPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.flaring_pipeline import PIPELINE_STAGES
except ImportError:
    PIPELINE_STAGES = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and response models
# ---------------------------------------------------------------------------
try:
    from greenlang.flaring.setup import (
        FlaringService,
        configure_flaring,
        get_service,
        get_router,
        CalculateResponse,
        BatchCalculateResponse,
        FlareSystemResponse,
        FlareSystemListResponse,
        FlaringEventResponse,
        FlaringEventListResponse,
        GasCompositionResponse,
        GasCompositionListResponse,
        EmissionFactorResponse,
        EmissionFactorListResponse,
        EfficiencyTestResponse,
        EfficiencyTestListResponse,
        UncertaintyResponse,
        ComplianceCheckResponse,
        StatsResponse,
    )
except ImportError:
    FlaringService = None  # type: ignore[assignment, misc]
    configure_flaring = None  # type: ignore[assignment, misc]
    get_service = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]
    CalculateResponse = None  # type: ignore[assignment, misc]
    BatchCalculateResponse = None  # type: ignore[assignment, misc]
    FlareSystemResponse = None  # type: ignore[assignment, misc]
    FlareSystemListResponse = None  # type: ignore[assignment, misc]
    FlaringEventResponse = None  # type: ignore[assignment, misc]
    FlaringEventListResponse = None  # type: ignore[assignment, misc]
    GasCompositionResponse = None  # type: ignore[assignment, misc]
    GasCompositionListResponse = None  # type: ignore[assignment, misc]
    EmissionFactorResponse = None  # type: ignore[assignment, misc]
    EmissionFactorListResponse = None  # type: ignore[assignment, misc]
    EfficiencyTestResponse = None  # type: ignore[assignment, misc]
    EfficiencyTestListResponse = None  # type: ignore[assignment, misc]
    UncertaintyResponse = None  # type: ignore[assignment, misc]
    ComplianceCheckResponse = None  # type: ignore[assignment, misc]
    StatsResponse = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Config standard condition constants re-export
# ---------------------------------------------------------------------------
from greenlang.flaring.config import (
    EPA_STANDARD_TEMP_F,
    EPA_STANDARD_TEMP_C,
    EPA_STANDARD_PRESSURE_PSIA,
    EPA_STANDARD_PRESSURE_KPA,
    ISO_STANDARD_TEMP_C,
    ISO_STANDARD_TEMP_F,
    ISO_STANDARD_PRESSURE_KPA,
    ISO_STANDARD_PRESSURE_PSIA,
    DEFAULT_HHV_CH4_BTU_SCF,
    DEFAULT_HHV_C2H6_BTU_SCF,
    DEFAULT_HHV_C3H8_BTU_SCF,
    DEFAULT_HHV_NC4H10_BTU_SCF,
    DEFAULT_HHV_IC4H10_BTU_SCF,
    DEFAULT_HHV_C5H12_BTU_SCF,
    DEFAULT_HHV_C6PLUS_BTU_SCF,
    DEFAULT_HHV_H2_BTU_SCF,
    DEFAULT_HHV_CO_BTU_SCF,
    DEFAULT_HHV_C2H4_BTU_SCF,
    DEFAULT_HHV_C3H6_BTU_SCF,
    SPEED_OF_SOUND_MS,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "FLARING_SDK_AVAILABLE",
    # Configuration
    "FlaringConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceEntry",
    "ProvenanceChain",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "VALID_STEP_NAMES",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # MetricsCollector class
    "MetricsCollector",
    # Metric objects
    "fl_calculations_total",
    "fl_emissions_kg_co2e_total",
    "fl_flare_lookups_total",
    "fl_factor_selections_total",
    "fl_flaring_events_total",
    "fl_uncertainty_runs_total",
    "fl_compliance_checks_total",
    "fl_batch_jobs_total",
    "fl_calculation_duration_seconds",
    "fl_batch_size",
    "fl_active_calculations",
    "fl_flare_systems_registered",
    # Metric helper functions
    "record_calculation",
    "record_emissions",
    "record_flare_lookup",
    "record_factor_selection",
    "record_event",
    "record_uncertainty",
    "record_compliance",
    "record_batch",
    "observe_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_flare_systems_registered",
    # Constants / lookup tables
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_COMPOSITION_COMPONENTS",
    "CO2_MOLECULAR_WEIGHT",
    "CARBON_ATOMIC_WEIGHT",
    "GWP_VALUES",
    "COMPONENT_HHV_BTU_SCF",
    "COMPONENT_MOLECULAR_WEIGHTS",
    "COMPONENT_CARBON_COUNT",
    "DEFAULT_COMBUSTION_EFFICIENCY",
    # Enumerations (16)
    "FlareType",
    "FlaringEventCategory",
    "CalculationMethod",
    "EmissionFactorSource",
    "GasComponent",
    "EmissionGas",
    "GWPSource",
    "StandardCondition",
    "AssistType",
    "FlaringStatus",
    "OGMPLevel",
    "ComplianceFramework",
    "CalculationStatus",
    "DataQualityTier",
    "SeverityLevel",
    "ComplianceStatus",
    # Data models (16+)
    "GasComposition",
    "FlareSystemConfig",
    "FlaringEventRecord",
    "PilotPurgeConfig",
    "CombustionEfficiencyParams",
    "CalculationInput",
    "CalculationResult",
    "EmissionDetail",
    "BatchCalculationRequest",
    "BatchCalculationResponse",
    "UncertaintyInput",
    "UncertaintyResult",
    "ComplianceCheckInput",
    "ComplianceCheckResult",
    "FlareSystemRegistration",
    "FlaringStats",
    "HealthResponse",
    # Core engines (Layer 2)
    "FlareSystemDatabaseEngine",
    "FlaringEventTrackerEngine",
    "UncertaintyQuantifierEngine",
    "FlaringPipelineEngine",
    # Pipeline stages
    "PIPELINE_STAGES",
    # Service setup facade
    "FlaringService",
    "configure_flaring",
    "get_service",
    "get_router",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "FlareSystemResponse",
    "FlareSystemListResponse",
    "FlaringEventResponse",
    "FlaringEventListResponse",
    "GasCompositionResponse",
    "GasCompositionListResponse",
    "EmissionFactorResponse",
    "EmissionFactorListResponse",
    "EfficiencyTestResponse",
    "EfficiencyTestListResponse",
    "UncertaintyResponse",
    "ComplianceCheckResponse",
    "StatsResponse",
    # Standard condition constants (from config)
    "EPA_STANDARD_TEMP_F",
    "EPA_STANDARD_TEMP_C",
    "EPA_STANDARD_PRESSURE_PSIA",
    "EPA_STANDARD_PRESSURE_KPA",
    "ISO_STANDARD_TEMP_C",
    "ISO_STANDARD_TEMP_F",
    "ISO_STANDARD_PRESSURE_KPA",
    "ISO_STANDARD_PRESSURE_PSIA",
    # Default HHV constants (from config)
    "DEFAULT_HHV_CH4_BTU_SCF",
    "DEFAULT_HHV_C2H6_BTU_SCF",
    "DEFAULT_HHV_C3H8_BTU_SCF",
    "DEFAULT_HHV_NC4H10_BTU_SCF",
    "DEFAULT_HHV_IC4H10_BTU_SCF",
    "DEFAULT_HHV_C5H12_BTU_SCF",
    "DEFAULT_HHV_C6PLUS_BTU_SCF",
    "DEFAULT_HHV_H2_BTU_SCF",
    "DEFAULT_HHV_CO_BTU_SCF",
    "DEFAULT_HHV_C2H4_BTU_SCF",
    "DEFAULT_HHV_C3H6_BTU_SCF",
    # Speed of sound (from config)
    "SPEED_OF_SOUND_MS",
]
