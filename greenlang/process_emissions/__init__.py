# -*- coding: utf-8 -*-
"""
GL-MRV-SCOPE1-004: GreenLang Process Emissions Agent Service SDK
=================================================================

This package provides Scope 1 GHG emissions calculation from non-combustion
industrial process sources for the GreenLang framework. It supports:

- 25 industrial process types across 6 categories (mineral, chemical, metal,
  electronics, pulp & paper, other) covering cement, lime, glass, ammonia,
  nitric acid, adipic acid, iron & steel, aluminum smelting, and more
- Multi-gas tracking (CO2, CH4, N2O, CF4, C2F6, SF6, NF3, HFC) with
  process-specific gas profiles
- 4 calculation methods: emission factor, mass balance, stoichiometric,
  and direct measurement
- Tier 1/2/3 methodology per IPCC 2006 Guidelines Chapter 2-4
- Carbonate decomposition factors (calcite, dolomite, magnesite, siderite,
  ankerite) for mineral process emissions
- GWP application from AR4, AR5, AR6, and AR6-20yr timeframes for all
  8 greenhouse gas species including fluorinated gases
- Production route tracking for iron/steel (BF-BOF, EAF, DRI) and
  aluminum (prebake, Soderberg VSS/HSS) processes
- Abatement tracking (catalytic reduction, thermal destruction, scrubbing,
  carbon capture, PFC anode control, SF6 recovery, SCR/NSCR)
- By-product emission credits for processes with co-products
- Monte Carlo uncertainty quantification (configurable iterations) with
  analytical error propagation and data quality scoring
- Multi-framework regulatory compliance checking (GHG Protocol, ISO 14064,
  CSRD/ESRS E1, EPA 40 CFR Part 98, UK SECR, EU ETS)
- Complete SHA-256 provenance chain tracking for audit trails
- 12 Prometheus metrics with gl_pe_ prefix for observability
- Thread-safe configuration with GL_PROCESS_EMISSIONS_ env prefix

Key Components:
    - config: ProcessEmissionsConfig with GL_PROCESS_EMISSIONS_ env prefix
    - models: Pydantic v2 models (16 enums, 16+ data models, GWP/carbonate
      constant tables)
    - metrics: 12 Prometheus metrics with gl_pe_ prefix
    - provenance: SHA-256 chain-hashed audit trails with 10 entity types
      and 15 actions

Example:
    >>> from greenlang.process_emissions import ProcessEmissionsConfig
    >>> from greenlang.process_emissions import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_gwp_source, cfg.default_calculation_tier)
    AR6 TIER_1

Agent ID: GL-MRV-SCOPE1-004
Agent Name: Process Emissions Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-SCOPE1-004"
__agent_name__ = "Process Emissions Agent"

# SDK availability flag
PROCESS_EMISSIONS_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.process_emissions.config import (
    ProcessEmissionsConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.process_emissions.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
try:
    from greenlang.process_emissions.metrics import (
        PROMETHEUS_AVAILABLE,
        pe_calculations_total,
        pe_emissions_kg_co2e_total,
        pe_process_lookups_total,
        pe_factor_selections_total,
        pe_material_operations_total,
        pe_uncertainty_runs_total,
        pe_compliance_checks_total,
        pe_batch_jobs_total,
        pe_calculation_duration_seconds,
        pe_batch_size,
        pe_active_calculations,
        pe_process_units_registered,
        record_calculation,
        record_emissions,
        record_process_lookup,
        record_factor_selection,
        record_material_operation,
        record_uncertainty,
        record_compliance_check,
        record_batch,
        observe_calculation_duration,
        observe_batch_size,
        set_active_calculations,
        set_process_units_registered,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    pe_calculations_total = None  # type: ignore[assignment]
    pe_emissions_kg_co2e_total = None  # type: ignore[assignment]
    pe_process_lookups_total = None  # type: ignore[assignment]
    pe_factor_selections_total = None  # type: ignore[assignment]
    pe_material_operations_total = None  # type: ignore[assignment]
    pe_uncertainty_runs_total = None  # type: ignore[assignment]
    pe_compliance_checks_total = None  # type: ignore[assignment]
    pe_batch_jobs_total = None  # type: ignore[assignment]
    pe_calculation_duration_seconds = None  # type: ignore[assignment]
    pe_batch_size = None  # type: ignore[assignment]
    pe_active_calculations = None  # type: ignore[assignment]
    pe_process_units_registered = None  # type: ignore[assignment]
    record_calculation = None  # type: ignore[assignment]
    record_emissions = None  # type: ignore[assignment]
    record_process_lookup = None  # type: ignore[assignment]
    record_factor_selection = None  # type: ignore[assignment]
    record_material_operation = None  # type: ignore[assignment]
    record_uncertainty = None  # type: ignore[assignment]
    record_compliance_check = None  # type: ignore[assignment]
    record_batch = None  # type: ignore[assignment]
    observe_calculation_duration = None  # type: ignore[assignment]
    observe_batch_size = None  # type: ignore[assignment]
    set_active_calculations = None  # type: ignore[assignment]
    set_process_units_registered = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Models (enums, constants, data models) - optional import with graceful
# fallback so that the SDK can still be imported when pydantic is absent
# ---------------------------------------------------------------------------
try:
    from greenlang.process_emissions.models import (
        # Constants / lookup tables
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        MAX_GASES_PER_RESULT,
        MAX_TRACE_STEPS,
        MAX_MATERIAL_INPUTS_PER_CALC,
        GWP_VALUES,
        CARBONATE_EMISSION_FACTORS,
        # Enumerations
        ProcessCategory,
        ProcessType,
        EmissionGas,
        CalculationMethod,
        CalculationTier,
        EmissionFactorSource,
        GWPSource,
        MaterialType,
        AbatementType,
        ProcessUnitType,
        ProcessMode,
        ComplianceStatus,
        ReportingPeriod,
        UnitType,
        ProductionRoute,
        CarbonateType,
        # Data models
        ProcessTypeInfo,
        RawMaterialInfo,
        EmissionFactorRecord,
        ProcessUnitRecord,
        MaterialInputRecord,
        CalculationRequest,
        GasEmissionResult,
        CalculationResult,
        CalculationDetailResult,
        AbatementRecord,
        ComplianceCheckResult,
        BatchCalculationRequest,
        BatchCalculationResult,
        UncertaintyRequest,
        UncertaintyResult,
        AggregationRequest,
        AggregationResult,
    )
except ImportError:
    # Constants
    VERSION = None  # type: ignore[assignment]
    MAX_CALCULATIONS_PER_BATCH = None  # type: ignore[assignment]
    MAX_GASES_PER_RESULT = None  # type: ignore[assignment]
    MAX_TRACE_STEPS = None  # type: ignore[assignment]
    MAX_MATERIAL_INPUTS_PER_CALC = None  # type: ignore[assignment]
    GWP_VALUES = None  # type: ignore[assignment]
    CARBONATE_EMISSION_FACTORS = None  # type: ignore[assignment]
    # Enumerations
    ProcessCategory = None  # type: ignore[assignment, misc]
    ProcessType = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationTier = None  # type: ignore[assignment, misc]
    EmissionFactorSource = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    MaterialType = None  # type: ignore[assignment, misc]
    AbatementType = None  # type: ignore[assignment, misc]
    ProcessUnitType = None  # type: ignore[assignment, misc]
    ProcessMode = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    UnitType = None  # type: ignore[assignment, misc]
    ProductionRoute = None  # type: ignore[assignment, misc]
    CarbonateType = None  # type: ignore[assignment, misc]
    # Data models
    ProcessTypeInfo = None  # type: ignore[assignment, misc]
    RawMaterialInfo = None  # type: ignore[assignment, misc]
    EmissionFactorRecord = None  # type: ignore[assignment, misc]
    ProcessUnitRecord = None  # type: ignore[assignment, misc]
    MaterialInputRecord = None  # type: ignore[assignment, misc]
    CalculationRequest = None  # type: ignore[assignment, misc]
    GasEmissionResult = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    CalculationDetailResult = None  # type: ignore[assignment, misc]
    AbatementRecord = None  # type: ignore[assignment, misc]
    ComplianceCheckResult = None  # type: ignore[assignment, misc]
    BatchCalculationRequest = None  # type: ignore[assignment, misc]
    BatchCalculationResult = None  # type: ignore[assignment, misc]
    UncertaintyRequest = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    AggregationRequest = None  # type: ignore[assignment, misc]
    AggregationResult = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "PROCESS_EMISSIONS_SDK_AVAILABLE",
    # Configuration
    "ProcessEmissionsConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceEntry",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "pe_calculations_total",
    "pe_emissions_kg_co2e_total",
    "pe_process_lookups_total",
    "pe_factor_selections_total",
    "pe_material_operations_total",
    "pe_uncertainty_runs_total",
    "pe_compliance_checks_total",
    "pe_batch_jobs_total",
    "pe_calculation_duration_seconds",
    "pe_batch_size",
    "pe_active_calculations",
    "pe_process_units_registered",
    # Metric helper functions
    "record_calculation",
    "record_emissions",
    "record_process_lookup",
    "record_factor_selection",
    "record_material_operation",
    "record_uncertainty",
    "record_compliance_check",
    "record_batch",
    "observe_calculation_duration",
    "observe_batch_size",
    "set_active_calculations",
    "set_process_units_registered",
    # Constants / lookup tables
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_MATERIAL_INPUTS_PER_CALC",
    "GWP_VALUES",
    "CARBONATE_EMISSION_FACTORS",
    # Enumerations
    "ProcessCategory",
    "ProcessType",
    "EmissionGas",
    "CalculationMethod",
    "CalculationTier",
    "EmissionFactorSource",
    "GWPSource",
    "MaterialType",
    "AbatementType",
    "ProcessUnitType",
    "ProcessMode",
    "ComplianceStatus",
    "ReportingPeriod",
    "UnitType",
    "ProductionRoute",
    "CarbonateType",
    # Data models
    "ProcessTypeInfo",
    "RawMaterialInfo",
    "EmissionFactorRecord",
    "ProcessUnitRecord",
    "MaterialInputRecord",
    "CalculationRequest",
    "GasEmissionResult",
    "CalculationResult",
    "CalculationDetailResult",
    "AbatementRecord",
    "ComplianceCheckResult",
    "BatchCalculationRequest",
    "BatchCalculationResult",
    "UncertaintyRequest",
    "UncertaintyResult",
    "AggregationRequest",
    "AggregationResult",
]
