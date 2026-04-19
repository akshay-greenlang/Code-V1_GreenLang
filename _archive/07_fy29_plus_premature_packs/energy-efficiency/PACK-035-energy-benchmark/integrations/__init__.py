# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark Pack - Integration Layer
======================================================

Phase 4 integration layer for the Energy Benchmark Pack that provides
benchmark pipeline orchestration, MRV carbon intensity bridging, DATA agent
routing, PACK-031/032/033 data imports, ENERGY STAR score calculation,
weather service data for normalisation, EPC registry lookups, benchmark
database unified access, 15-category health verification, and 8-step
facility setup wizard.

Components:
    - EnergyBenchmarkOrchestrator: 12-phase pipeline with DAG dependency
      resolution, parallel execution, retry with exponential backoff,
      and SHA-256 provenance tracking
    - MRVBenchmarkBridge: Routes emission factor data from MRV agents
      (Stationary Combustion, Scope 2 Location/Market, Dual Reporting)
      for carbon intensity benchmarking
    - DataBenchmarkBridge: Routes data intake to DATA agents for meter
      data, utility bills, quality profiling, outlier detection, gap
      filling, and data lineage tracking
    - Pack031Bridge: Imports energy audit baselines, EnPI data, and
      equipment efficiency from PACK-031 Industrial Energy Audit
    - Pack032Bridge: Imports building assessment data, zone breakdowns,
      and envelope characteristics from PACK-032
    - Pack033Bridge: Imports quick win measures and links benchmark gap
      analysis to corrective actions from PACK-033
    - EnergyStarBridge: ENERGY STAR Portfolio Manager score calculation,
      50+ property types, source EUI lookup tables
    - WeatherServiceBridge: Weather station data, HDD/CDD, TMY data
      from NOAA, Meteostat, CIBSE TRY, DWD, ASHRAE IWEC
    - EPCRegistryBridge: EPC certificate lookup from UK and EU registries
    - BenchmarkDatabaseBridge: Unified interface to CIBSE TM46, DIN V
      18599, BPIE, and national benchmark databases
    - HealthCheck: 15-category system health verification
    - SetupWizard: 8-step guided facility configuration

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    PACK-035 Benchmark <-- Composition <-- Zero Duplication
                              |
                              v
    MRV Agents <-- DATA Agents <-- FOUND Agents

Platform Integrations:
    - greenlang/agents/mrv/* (MRV-001, 009, 010, 013)
    - greenlang/agents/data/* (DATA-001, 002, 004, 010, 013, 014, 018)
    - packs/energy-efficiency/PACK-031 (baselines, EnPI)
    - packs/energy-efficiency/PACK-032 (building assessment)
    - packs/energy-efficiency/PACK-033 (quick wins)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-035 Energy Benchmark
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-035"
__pack_name__ = "Energy Benchmark Pack"

# ---------------------------------------------------------------------------
# Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.pack_orchestrator import (
    BuildingSector,
    EnergyBenchmarkOrchestrator,
    OrchestratorConfig,
    OrchestratorPhase,
    PARALLEL_PHASE_GROUPS,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PhaseStatus,
    PipelineResult,
    RetryConfig,
    RetryStrategy,
)

# ---------------------------------------------------------------------------
# MRV Benchmark Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.mrv_benchmark_bridge import (
    CarbonIntensityMetric,
    CarbonIntensityResult,
    EmissionFactorRequest,
    EmissionFactorResult,
    EmissionScope,
    MRVBenchmarkBridge,
    MRVBenchmarkBridgeConfig,
)

# ---------------------------------------------------------------------------
# Data Benchmark Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.data_benchmark_bridge import (
    BenchmarkDataSource,
    DataBenchmarkBridge,
    DataBenchmarkBridgeConfig,
    DataIngestionRequest,
    DataIngestionResult,
    DataQualityDimension,
)

# ---------------------------------------------------------------------------
# PACK-031 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.pack031_bridge import (
    AuditBaselineRequest,
    AuditBaselineResult,
    Pack031Bridge,
    Pack031BridgeConfig,
)

# ---------------------------------------------------------------------------
# PACK-032 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.pack032_bridge import (
    BuildingAssessmentRequest,
    BuildingAssessmentResult,
    Pack032Bridge,
    Pack032BridgeConfig,
)

# ---------------------------------------------------------------------------
# PACK-033 Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.pack033_bridge import (
    GapMeasureLink,
    Pack033Bridge,
    Pack033BridgeConfig,
    QuickWinRequest,
    QuickWinResult,
)

# ---------------------------------------------------------------------------
# ENERGY STAR Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.energy_star_bridge import (
    EnergyStarBridge,
    EnergyStarBridgeConfig,
    EnergyStarMetric,
    EnergyStarPropertyType,
    EnergyStarScoreResult,
    PropertyInput,
)

# ---------------------------------------------------------------------------
# Weather Service Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.weather_service_bridge import (
    DegreeDayMethod,
    DegreeDayRequest,
    TMYRequest,
    WeatherDataResult,
    WeatherDataSource,
    WeatherServiceBridge,
    WeatherServiceConfig,
    WeatherStationInfo,
)

# ---------------------------------------------------------------------------
# EPC Registry Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.epc_registry_bridge import (
    EPCCertificateData,
    EPCLookupRequest,
    EPCRating,
    EPCRegistryBridge,
    EPCRegistryConfig,
    EPCRegistryRegion,
)

# ---------------------------------------------------------------------------
# Benchmark Database Bridge
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.benchmark_database_bridge import (
    BenchmarkDatabaseBridge,
    BenchmarkDatabaseConfig,
    BenchmarkDatabaseResult,
    BenchmarkDatabaseSource,
    BenchmarkQuery,
    BuildingClassification,
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.health_check import (
    CategoryHealth,
    HealthCategory,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
)

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_035_energy_benchmark.integrations.setup_wizard import (
    FacilitySetup,
    FacilityType,
    SetupResult,
    SetupWizard,
    SetupWizardConfig,
    StepResult,
    StepStatus,
    WizardState,
    WizardStep,
    WizardStepState,
)

__all__ = [
    # Module metadata
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pack Orchestrator ---
    "EnergyBenchmarkOrchestrator",
    "OrchestratorConfig",
    "RetryConfig",
    "OrchestratorPhase",
    "PhaseStatus",
    "RetryStrategy",
    "BuildingSector",
    "PhaseProvenance",
    "PhaseResult",
    "PipelineResult",
    "PHASE_DEPENDENCIES",
    "PHASE_EXECUTION_ORDER",
    "PARALLEL_PHASE_GROUPS",
    # --- MRV Benchmark Bridge ---
    "MRVBenchmarkBridge",
    "MRVBenchmarkBridgeConfig",
    "EmissionScope",
    "CarbonIntensityMetric",
    "EmissionFactorRequest",
    "EmissionFactorResult",
    "CarbonIntensityResult",
    # --- Data Benchmark Bridge ---
    "DataBenchmarkBridge",
    "DataBenchmarkBridgeConfig",
    "BenchmarkDataSource",
    "DataQualityDimension",
    "DataIngestionRequest",
    "DataIngestionResult",
    # --- PACK-031 Bridge ---
    "Pack031Bridge",
    "Pack031BridgeConfig",
    "AuditBaselineRequest",
    "AuditBaselineResult",
    # --- PACK-032 Bridge ---
    "Pack032Bridge",
    "Pack032BridgeConfig",
    "BuildingAssessmentRequest",
    "BuildingAssessmentResult",
    # --- PACK-033 Bridge ---
    "Pack033Bridge",
    "Pack033BridgeConfig",
    "QuickWinRequest",
    "QuickWinResult",
    "GapMeasureLink",
    # --- ENERGY STAR Bridge ---
    "EnergyStarBridge",
    "EnergyStarBridgeConfig",
    "EnergyStarPropertyType",
    "EnergyStarMetric",
    "PropertyInput",
    "EnergyStarScoreResult",
    # --- Weather Service Bridge ---
    "WeatherServiceBridge",
    "WeatherServiceConfig",
    "WeatherDataSource",
    "DegreeDayMethod",
    "WeatherStationInfo",
    "DegreeDayRequest",
    "TMYRequest",
    "WeatherDataResult",
    # --- EPC Registry Bridge ---
    "EPCRegistryBridge",
    "EPCRegistryConfig",
    "EPCRegistryRegion",
    "EPCRating",
    "EPCLookupRequest",
    "EPCCertificateData",
    # --- Benchmark Database Bridge ---
    "BenchmarkDatabaseBridge",
    "BenchmarkDatabaseConfig",
    "BenchmarkDatabaseSource",
    "BuildingClassification",
    "BenchmarkQuery",
    "BenchmarkDatabaseResult",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "CategoryHealth",
    "HealthCategory",
    "HealthStatus",
    "HealthSeverity",
    # --- Setup Wizard ---
    "SetupWizard",
    "SetupWizardConfig",
    "WizardStep",
    "StepStatus",
    "FacilityType",
    "FacilitySetup",
    "WizardStepState",
    "WizardState",
    "StepResult",
    "SetupResult",
]
