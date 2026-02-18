# -*- coding: utf-8 -*-
"""
GL-MRV-SCOPE1-001: GreenLang Stationary Combustion Agent Service SDK
====================================================================

This package provides Scope 1 GHG emissions calculation from stationary
combustion sources for the GreenLang framework. It supports:

- Multi-source fuel database (EPA, IPCC, DEFRA, EU ETS, custom) with
  327+ emission factors for 24 fuel types across gaseous, liquid, solid,
  and biomass categories
- Tier 1/2/3 emission factor selection with automatic fallback chain
  (Tier 3 facility-specific -> Tier 2 country-specific -> Tier 1 IPCC default)
- Core combustion calculation: Activity x HV x EF x OF x GWP with
  Decimal-precision arithmetic and gas decomposition (CO2, CH4, N2O)
- Equipment profiling with efficiency curves, load factors, age degradation
  for 13 equipment types (boilers, furnaces, turbines, kilns, etc.)
- Monte Carlo uncertainty quantification (5000+ iterations) with analytical
  error propagation and data quality scoring (1-5 scale)
- Biogenic carbon tracking (wood, biomass, biogas, landfill gas) with
  separate reporting per GHG Protocol guidance
- Multi-framework regulatory compliance mapping (GHG Protocol, ISO 14064-1,
  CSRD/ESRS E1, EPA 40 CFR Part 98 Subpart C, UK SECR, EU ETS MRR)
- HHV/NCV heating value conversion with fuel-specific ratios
- GWP application (AR4, AR5, AR6, 20yr/100yr timeframes)
- Facility aggregation with operational/financial/equity share control
- Complete SHA-256 provenance chain tracking for audit trails
- 12 Prometheus metrics with gl_sc_ prefix for observability
- FastAPI REST API with 20 endpoints at /api/v1/stationary-combustion
- Thread-safe configuration with GL_STATIONARY_COMBUSTION_ env prefix
- 7-engine architecture: FuelDatabaseEngine, CombustionCalculatorEngine,
  EquipmentProfilerEngine, EmissionFactorSelectorEngine,
  UncertaintyQuantifierEngine, AuditTrailEngine,
  StationaryCombustionPipelineEngine

Key Components:
    - config: StationaryCombustionConfig with GL_STATIONARY_COMBUSTION_ env prefix
    - models: Pydantic v2 models (13 enums, 12+ data models)
    - fuel_database: Fuel type registry and emission factor database engine
    - combustion_calculator: Core combustion emissions calculation engine
    - equipment_profiler: Equipment type profiling and efficiency engine
    - emission_factor_selector: Tier-based emission factor selection engine
    - uncertainty_quantifier: Monte Carlo and analytical uncertainty engine
    - audit_trail: Calculation lineage and compliance mapping engine
    - combustion_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_sc_ prefix
    - api: FastAPI HTTP service with 20 endpoints
    - setup: StationaryCombustionService facade

Example:
    >>> from greenlang.stationary_combustion import StationaryCombustionService
    >>> service = StationaryCombustionService()
    >>> result = service.calculate(
    ...     fuel_type="NATURAL_GAS",
    ...     quantity=1000.0,
    ...     unit="CUBIC_METERS",
    ... )
    >>> print(result["total_co2e_tonnes"])

Agent ID: GL-MRV-SCOPE1-001
Agent Name: Stationary Combustion Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE1-001"
__agent_name__ = "Stationary Combustion Agent"

# SDK availability flag
STATIONARY_COMBUSTION_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.stationary_combustion.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    get_provenance_tracker,
    reset_provenance_tracker,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
try:
    from greenlang.stationary_combustion.metrics import (
        PROMETHEUS_AVAILABLE,
        sc_calculations_total,
        sc_emissions_kg_co2e_total,
        sc_fuel_lookups_total,
        sc_factor_selections_total,
        sc_equipment_profiles_total,
        sc_uncertainty_runs_total,
        sc_audit_entries_total,
        sc_batch_jobs_total,
        sc_calculation_duration_seconds,
        sc_batch_size,
        sc_active_calculations,
        sc_emission_factors_loaded,
        record_calculation,
        record_emissions,
        record_fuel_lookup,
        record_factor_selection,
        record_equipment,
        record_uncertainty,
        record_audit,
        record_batch,
        observe_calculation_duration,
        observe_batch_size,
        set_active_calculations,
        set_factors_loaded,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    sc_calculations_total = None  # type: ignore[assignment]
    sc_emissions_kg_co2e_total = None  # type: ignore[assignment]
    sc_fuel_lookups_total = None  # type: ignore[assignment]
    sc_factor_selections_total = None  # type: ignore[assignment]
    sc_equipment_profiles_total = None  # type: ignore[assignment]
    sc_uncertainty_runs_total = None  # type: ignore[assignment]
    sc_audit_entries_total = None  # type: ignore[assignment]
    sc_batch_jobs_total = None  # type: ignore[assignment]
    sc_calculation_duration_seconds = None  # type: ignore[assignment]
    sc_batch_size = None  # type: ignore[assignment]
    sc_active_calculations = None  # type: ignore[assignment]
    sc_emission_factors_loaded = None  # type: ignore[assignment]
    record_calculation = None  # type: ignore[assignment]
    record_emissions = None  # type: ignore[assignment]
    record_fuel_lookup = None  # type: ignore[assignment]
    record_factor_selection = None  # type: ignore[assignment]
    record_equipment = None  # type: ignore[assignment]
    record_uncertainty = None  # type: ignore[assignment]
    record_audit = None  # type: ignore[assignment]
    record_batch = None  # type: ignore[assignment]
    observe_calculation_duration = None  # type: ignore[assignment]
    observe_batch_size = None  # type: ignore[assignment]
    set_active_calculations = None  # type: ignore[assignment]
    set_factors_loaded = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Models (enums, data models) - optional import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.stationary_combustion.models import (
        # Enumerations
        FuelCategory,
        FuelType,
        EmissionGas,
        GWPSource,
        EFSource,
        CalculationTier,
        EquipmentType,
        HeatingValueBasis,
        ControlApproach,
        CalculationStatus,
        ReportingPeriod,
        RegulatoryFramework,
        UnitType,
        # Core data models
        EmissionFactor,
        FuelProperties,
        EquipmentProfile,
        CombustionInput,
        GasEmission,
        CalculationResult,
        BatchCalculationRequest,
        BatchCalculationResponse,
        UncertaintyResult,
        FacilityAggregation,
        AuditEntry,
        ComplianceMapping,
    )
except ImportError:
    FuelCategory = None  # type: ignore[assignment, misc]
    FuelType = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    EFSource = None  # type: ignore[assignment, misc]
    CalculationTier = None  # type: ignore[assignment, misc]
    EquipmentType = None  # type: ignore[assignment, misc]
    HeatingValueBasis = None  # type: ignore[assignment, misc]
    ControlApproach = None  # type: ignore[assignment, misc]
    CalculationStatus = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    RegulatoryFramework = None  # type: ignore[assignment, misc]
    UnitType = None  # type: ignore[assignment, misc]
    EmissionFactor = None  # type: ignore[assignment, misc]
    FuelProperties = None  # type: ignore[assignment, misc]
    EquipmentProfile = None  # type: ignore[assignment, misc]
    CombustionInput = None  # type: ignore[assignment, misc]
    GasEmission = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    BatchCalculationRequest = None  # type: ignore[assignment, misc]
    BatchCalculationResponse = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    FacilityAggregation = None  # type: ignore[assignment, misc]
    AuditEntry = None  # type: ignore[assignment, misc]
    ComplianceMapping = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
except ImportError:
    FuelDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.combustion_calculator import CombustionCalculatorEngine
except ImportError:
    CombustionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.equipment_profiler import (
        EquipmentProfilerEngine,
        EQUIPMENT_DEFAULTS,
    )
except ImportError:
    EquipmentProfilerEngine = None  # type: ignore[assignment, misc]
    EQUIPMENT_DEFAULTS = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.emission_factor_selector import EmissionFactorSelectorEngine
except ImportError:
    EmissionFactorSelectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.audit_trail import AuditTrailEngine
except ImportError:
    AuditTrailEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.combustion_pipeline import StationaryCombustionPipelineEngine
except ImportError:
    StationaryCombustionPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.combustion_pipeline import PIPELINE_STAGES
except ImportError:
    PIPELINE_STAGES = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and response models
# ---------------------------------------------------------------------------
try:
    from greenlang.stationary_combustion.setup import (
        StationaryCombustionService,
        configure_stationary_combustion,
        get_service,
        get_router,
        CalculationResponse,
        BatchResponse,
        FuelResponse,
        FuelListResponse,
        FactorResponse,
        EquipmentResponse,
        AggregationResponse,
        UncertaintyResponse,
        AuditTrailResponse,
        ComplianceResponse,
        ValidationResponse,
        PipelineResponse,
        HealthResponse,
        StatsResponse,
    )
except ImportError:
    StationaryCombustionService = None  # type: ignore[assignment, misc]
    configure_stationary_combustion = None  # type: ignore[assignment, misc]
    get_service = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]
    CalculationResponse = None  # type: ignore[assignment, misc]
    BatchResponse = None  # type: ignore[assignment, misc]
    FuelResponse = None  # type: ignore[assignment, misc]
    FuelListResponse = None  # type: ignore[assignment, misc]
    FactorResponse = None  # type: ignore[assignment, misc]
    EquipmentResponse = None  # type: ignore[assignment, misc]
    AggregationResponse = None  # type: ignore[assignment, misc]
    UncertaintyResponse = None  # type: ignore[assignment, misc]
    AuditTrailResponse = None  # type: ignore[assignment, misc]
    ComplianceResponse = None  # type: ignore[assignment, misc]
    ValidationResponse = None  # type: ignore[assignment, misc]
    PipelineResponse = None  # type: ignore[assignment, misc]
    HealthResponse = None  # type: ignore[assignment, misc]
    StatsResponse = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from mrv.scope1_combustion
# ---------------------------------------------------------------------------
try:
    from greenlang.agents.mrv.scope1_combustion import Scope1CombustionAgent
except ImportError:
    Scope1CombustionAgent = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "STATIONARY_COMBUSTION_SDK_AVAILABLE",
    # Configuration
    "StationaryCombustionConfig",
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
    "sc_calculations_total",
    "sc_emissions_kg_co2e_total",
    "sc_fuel_lookups_total",
    "sc_factor_selections_total",
    "sc_equipment_profiles_total",
    "sc_uncertainty_runs_total",
    "sc_audit_entries_total",
    "sc_batch_jobs_total",
    "sc_calculation_duration_seconds",
    "sc_batch_size",
    "sc_active_calculations",
    "sc_emission_factors_loaded",
    # Metric helper functions
    "record_calculation",
    "record_emissions",
    "record_fuel_lookup",
    "record_factor_selection",
    "record_equipment",
    "record_uncertainty",
    "record_audit",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_factors_loaded",
    # Enumerations
    "FuelCategory",
    "FuelType",
    "EmissionGas",
    "GWPSource",
    "EFSource",
    "CalculationTier",
    "EquipmentType",
    "HeatingValueBasis",
    "ControlApproach",
    "CalculationStatus",
    "ReportingPeriod",
    "RegulatoryFramework",
    "UnitType",
    # Core data models
    "EmissionFactor",
    "FuelProperties",
    "EquipmentProfile",
    "CombustionInput",
    "GasEmission",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResponse",
    "UncertaintyResult",
    "FacilityAggregation",
    "AuditEntry",
    "ComplianceMapping",
    # Core engines (Layer 2)
    "FuelDatabaseEngine",
    "CombustionCalculatorEngine",
    "EquipmentProfilerEngine",
    "EQUIPMENT_DEFAULTS",
    "EmissionFactorSelectorEngine",
    "UncertaintyQuantifierEngine",
    "AuditTrailEngine",
    "StationaryCombustionPipelineEngine",
    # Pipeline stages
    "PIPELINE_STAGES",
    # Service setup facade
    "StationaryCombustionService",
    "configure_stationary_combustion",
    "get_service",
    "get_router",
    # Response models
    "CalculationResponse",
    "BatchResponse",
    "FuelResponse",
    "FuelListResponse",
    "FactorResponse",
    "EquipmentResponse",
    "AggregationResponse",
    "UncertaintyResponse",
    "AuditTrailResponse",
    "ComplianceResponse",
    "ValidationResponse",
    "PipelineResponse",
    "HealthResponse",
    "StatsResponse",
    # Layer 1 re-export
    "Scope1CombustionAgent",
]
