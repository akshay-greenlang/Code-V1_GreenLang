# -*- coding: utf-8 -*-
"""
GL-MRV-SCOPE1-003: GreenLang Mobile Combustion Agent Service SDK
================================================================

This package provides Scope 1 GHG emissions calculation from mobile
combustion sources for the GreenLang framework. It supports:

- On-road vehicles (passenger cars, light/medium/heavy-duty trucks,
  buses, motorcycles, vans) with gasoline, diesel, hybrid, PHEV variants
- Off-road equipment (construction, agricultural, industrial, mining,
  forklifts) with diesel, gasoline, LPG fuel options
- Marine vessels (inland, coastal, ocean) burning marine diesel oil
  and heavy fuel oil
- Aviation sources (corporate jets, helicopters, turboprops) using
  jet fuel, avgas, and sustainable aviation fuel
- Rail (diesel locomotives) for freight and passenger transport
- Fuel-based, distance-based, and spend-based calculation methods per
  GHG Protocol Chapter 4 guidance
- 16 fuel types including biofuel blends (E10, E85, B5, B20, B100, SAF)
  with biogenic CO2 tracked separately per GHG Protocol
- Tier 1/2/3 emission factor selection with automatic fallback chain
  (Tier 3 model-year-specific -> Tier 2 vehicle-type-specific ->
  Tier 1 fuel-type default)
- 25 vehicle types across 5 categories (on-road, off-road, marine,
  aviation, rail)
- Fleet management with vehicle registration, trip tracking, and
  fleet-level aggregation with intensity metrics (kg CO2e/km, kg CO2e/L)
- Monte Carlo uncertainty quantification (5000+ iterations) with
  analytical error propagation and data quality scoring (1-5 scale)
- Multi-framework regulatory compliance mapping (GHG Protocol, ISO 14064,
  CSRD/ESRS E1, EPA 40 CFR Part 98, UK SECR, EU ETS)
- GWP application (AR4, AR5, AR6, AR6 20-year timeframes)
- Complete SHA-256 provenance chain tracking for audit trails
- 12 Prometheus metrics with gl_mc_ prefix for observability
- FastAPI REST API with endpoints at /api/v1/mobile-combustion
- Thread-safe configuration with GL_MOBILE_COMBUSTION_ env prefix
- 7-engine architecture: VehicleDatabaseEngine,
  CombustionCalculatorEngine, FleetManagerEngine,
  EmissionFactorSelectorEngine, UncertaintyQuantifierEngine,
  ComplianceCheckerEngine, MobileCombustionPipelineEngine

Key Components:
    - config: MobileCombustionConfig with GL_MOBILE_COMBUSTION_ env prefix
    - models: Pydantic v2 models (16 enums, 14+ data models)
    - vehicle_database: Vehicle type registry and fuel property database engine
    - combustion_calculator: Core mobile combustion emissions calculation engine
    - fleet_manager: Fleet vehicle registration and trip tracking engine
    - emission_factor_selector: Tier-based emission factor selection engine
    - uncertainty_quantifier: Monte Carlo and analytical uncertainty engine
    - compliance_checker: Regulatory compliance and framework mapping engine
    - combustion_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_mc_ prefix
    - api: FastAPI HTTP service with endpoints
    - setup: MobileCombustionService facade

Example:
    >>> from greenlang.mobile_combustion import MobileCombustionService
    >>> service = MobileCombustionService()
    >>> result = service.calculate(
    ...     vehicle_type="PASSENGER_CAR_GASOLINE",
    ...     fuel_type="GASOLINE",
    ...     quantity=100.0,
    ...     unit="LITERS",
    ... )
    >>> print(result["total_co2e_tonnes"])

Agent ID: GL-MRV-SCOPE1-003
Agent Name: Mobile Combustion Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE1-003"
__agent_name__ = "Mobile Combustion Agent"

# SDK availability flag
MOBILE_COMBUSTION_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.mobile_combustion.config import (
    MobileCombustionConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.mobile_combustion.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
try:
    from greenlang.mobile_combustion.metrics import (
        PROMETHEUS_AVAILABLE,
        mc_calculations_total,
        mc_emissions_kg_co2e_total,
        mc_vehicle_lookups_total,
        mc_factor_selections_total,
        mc_fleet_operations_total,
        mc_uncertainty_runs_total,
        mc_compliance_checks_total,
        mc_batch_jobs_total,
        mc_calculation_duration_seconds,
        mc_batch_size,
        mc_active_calculations,
        mc_vehicles_registered,
        record_calculation,
        record_emissions,
        record_vehicle_lookup,
        record_factor_selection,
        record_fleet_operation,
        record_uncertainty,
        record_compliance_check,
        record_batch,
        observe_calculation_duration,
        observe_batch_size,
        set_active_calculations,
        set_vehicles_registered,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    mc_calculations_total = None  # type: ignore[assignment]
    mc_emissions_kg_co2e_total = None  # type: ignore[assignment]
    mc_vehicle_lookups_total = None  # type: ignore[assignment]
    mc_factor_selections_total = None  # type: ignore[assignment]
    mc_fleet_operations_total = None  # type: ignore[assignment]
    mc_uncertainty_runs_total = None  # type: ignore[assignment]
    mc_compliance_checks_total = None  # type: ignore[assignment]
    mc_batch_jobs_total = None  # type: ignore[assignment]
    mc_calculation_duration_seconds = None  # type: ignore[assignment]
    mc_batch_size = None  # type: ignore[assignment]
    mc_active_calculations = None  # type: ignore[assignment]
    mc_vehicles_registered = None  # type: ignore[assignment]
    record_calculation = None  # type: ignore[assignment]
    record_emissions = None  # type: ignore[assignment]
    record_vehicle_lookup = None  # type: ignore[assignment]
    record_factor_selection = None  # type: ignore[assignment]
    record_fleet_operation = None  # type: ignore[assignment]
    record_uncertainty = None  # type: ignore[assignment]
    record_compliance_check = None  # type: ignore[assignment]
    record_batch = None  # type: ignore[assignment]
    observe_calculation_duration = None  # type: ignore[assignment]
    observe_batch_size = None  # type: ignore[assignment]
    set_active_calculations = None  # type: ignore[assignment]
    set_vehicles_registered = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Models (enums, data models) - optional import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.mobile_combustion.models import (
        # Constants
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        MAX_GASES_PER_RESULT,
        MAX_TRACE_STEPS,
        MAX_VEHICLES_PER_REGISTRATION,
        MAX_TRIPS_PER_BATCH,
        GWP_VALUES,
        BIOFUEL_FRACTIONS,
        # Enumerations
        VehicleCategory,
        VehicleType,
        FuelType,
        EmissionGas,
        CalculationMethod,
        CalculationTier,
        EmissionFactorSource,
        GWPSource,
        DistanceUnit,
        FuelEconomyUnit,
        EmissionControlTechnology,
        VehicleStatus,
        TripStatus,
        ComplianceStatus,
        ReportingPeriod,
        UnitType,
        # Core data models
        VehicleTypeInfo,
        FuelTypeInfo,
        EmissionFactorRecord,
        VehicleRegistration,
        TripRecord,
        CalculationInput,
        GasEmission,
        CalculationResult,
        BatchCalculationInput,
        BatchCalculationResponse,
        FleetAggregation,
        UncertaintyResult,
        ComplianceCheckResult,
        MobileCombustionInput,
        MobileCombustionOutput,
        AuditEntry,
    )
except ImportError:
    VERSION = None  # type: ignore[assignment]
    MAX_CALCULATIONS_PER_BATCH = None  # type: ignore[assignment]
    MAX_GASES_PER_RESULT = None  # type: ignore[assignment]
    MAX_TRACE_STEPS = None  # type: ignore[assignment]
    MAX_VEHICLES_PER_REGISTRATION = None  # type: ignore[assignment]
    MAX_TRIPS_PER_BATCH = None  # type: ignore[assignment]
    GWP_VALUES = None  # type: ignore[assignment]
    BIOFUEL_FRACTIONS = None  # type: ignore[assignment]
    VehicleCategory = None  # type: ignore[assignment, misc]
    VehicleType = None  # type: ignore[assignment, misc]
    FuelType = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationTier = None  # type: ignore[assignment, misc]
    EmissionFactorSource = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    DistanceUnit = None  # type: ignore[assignment, misc]
    FuelEconomyUnit = None  # type: ignore[assignment, misc]
    EmissionControlTechnology = None  # type: ignore[assignment, misc]
    VehicleStatus = None  # type: ignore[assignment, misc]
    TripStatus = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    UnitType = None  # type: ignore[assignment, misc]
    VehicleTypeInfo = None  # type: ignore[assignment, misc]
    FuelTypeInfo = None  # type: ignore[assignment, misc]
    EmissionFactorRecord = None  # type: ignore[assignment, misc]
    VehicleRegistration = None  # type: ignore[assignment, misc]
    TripRecord = None  # type: ignore[assignment, misc]
    CalculationInput = None  # type: ignore[assignment, misc]
    GasEmission = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    BatchCalculationInput = None  # type: ignore[assignment, misc]
    BatchCalculationResponse = None  # type: ignore[assignment, misc]
    FleetAggregation = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    ComplianceCheckResult = None  # type: ignore[assignment, misc]
    MobileCombustionInput = None  # type: ignore[assignment, misc]
    MobileCombustionOutput = None  # type: ignore[assignment, misc]
    AuditEntry = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
except ImportError:
    VehicleDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.combustion_calculator import CombustionCalculatorEngine
except ImportError:
    CombustionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.fleet_manager import FleetManagerEngine
except ImportError:
    FleetManagerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.emission_factor_selector import EmissionFactorSelectorEngine
except ImportError:
    EmissionFactorSelectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.combustion_pipeline import MobileCombustionPipelineEngine
except ImportError:
    MobileCombustionPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.combustion_pipeline import PIPELINE_STAGES
except ImportError:
    PIPELINE_STAGES = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and response models
# ---------------------------------------------------------------------------
try:
    from greenlang.mobile_combustion.setup import (
        MobileCombustionService,
        configure_mobile_combustion,
        get_service,
        get_router,
        CalculationResponse,
        BatchResponse,
        VehicleResponse,
        VehicleListResponse,
        TripResponse,
        TripListResponse,
        FactorResponse,
        FleetResponse,
        UncertaintyResponse,
        AuditTrailResponse,
        ComplianceResponse,
        PipelineResponse,
        HealthResponse,
        StatsResponse,
    )
except ImportError:
    MobileCombustionService = None  # type: ignore[assignment, misc]
    configure_mobile_combustion = None  # type: ignore[assignment, misc]
    get_service = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]
    CalculationResponse = None  # type: ignore[assignment, misc]
    BatchResponse = None  # type: ignore[assignment, misc]
    VehicleResponse = None  # type: ignore[assignment, misc]
    VehicleListResponse = None  # type: ignore[assignment, misc]
    TripResponse = None  # type: ignore[assignment, misc]
    TripListResponse = None  # type: ignore[assignment, misc]
    FactorResponse = None  # type: ignore[assignment, misc]
    FleetResponse = None  # type: ignore[assignment, misc]
    UncertaintyResponse = None  # type: ignore[assignment, misc]
    AuditTrailResponse = None  # type: ignore[assignment, misc]
    ComplianceResponse = None  # type: ignore[assignment, misc]
    PipelineResponse = None  # type: ignore[assignment, misc]
    HealthResponse = None  # type: ignore[assignment, misc]
    StatsResponse = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from mrv.scope1_mobile
# ---------------------------------------------------------------------------
try:
    from greenlang.agents.mrv.scope1_mobile import Scope1MobileCombustionAgent
except ImportError:
    Scope1MobileCombustionAgent = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "MOBILE_COMBUSTION_SDK_AVAILABLE",
    # Configuration
    "MobileCombustionConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceEntry",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "mc_calculations_total",
    "mc_emissions_kg_co2e_total",
    "mc_vehicle_lookups_total",
    "mc_factor_selections_total",
    "mc_fleet_operations_total",
    "mc_uncertainty_runs_total",
    "mc_compliance_checks_total",
    "mc_batch_jobs_total",
    "mc_calculation_duration_seconds",
    "mc_batch_size",
    "mc_active_calculations",
    "mc_vehicles_registered",
    # Metric helper functions
    "record_calculation",
    "record_emissions",
    "record_vehicle_lookup",
    "record_factor_selection",
    "record_fleet_operation",
    "record_uncertainty",
    "record_compliance_check",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_vehicles_registered",
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_VEHICLES_PER_REGISTRATION",
    "MAX_TRIPS_PER_BATCH",
    "GWP_VALUES",
    "BIOFUEL_FRACTIONS",
    # Enumerations
    "VehicleCategory",
    "VehicleType",
    "FuelType",
    "EmissionGas",
    "CalculationMethod",
    "CalculationTier",
    "EmissionFactorSource",
    "GWPSource",
    "DistanceUnit",
    "FuelEconomyUnit",
    "EmissionControlTechnology",
    "VehicleStatus",
    "TripStatus",
    "ComplianceStatus",
    "ReportingPeriod",
    "UnitType",
    # Core data models
    "VehicleTypeInfo",
    "FuelTypeInfo",
    "EmissionFactorRecord",
    "VehicleRegistration",
    "TripRecord",
    "CalculationInput",
    "GasEmission",
    "CalculationResult",
    "BatchCalculationInput",
    "BatchCalculationResponse",
    "FleetAggregation",
    "UncertaintyResult",
    "ComplianceCheckResult",
    "MobileCombustionInput",
    "MobileCombustionOutput",
    "AuditEntry",
    # Core engines (Layer 2)
    "VehicleDatabaseEngine",
    "CombustionCalculatorEngine",
    "FleetManagerEngine",
    "EmissionFactorSelectorEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "MobileCombustionPipelineEngine",
    # Pipeline stages
    "PIPELINE_STAGES",
    # Service setup facade
    "MobileCombustionService",
    "configure_mobile_combustion",
    "get_service",
    "get_router",
    # Response models
    "CalculationResponse",
    "BatchResponse",
    "VehicleResponse",
    "VehicleListResponse",
    "TripResponse",
    "TripListResponse",
    "FactorResponse",
    "FleetResponse",
    "UncertaintyResponse",
    "AuditTrailResponse",
    "ComplianceResponse",
    "PipelineResponse",
    "HealthResponse",
    "StatsResponse",
    # Layer 1 re-export
    "Scope1MobileCombustionAgent",
]
